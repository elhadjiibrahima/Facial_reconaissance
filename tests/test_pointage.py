"""Tests du pointage (sans caméra ni modèles)."""
import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pointage as pt  # noqa: E402

LUNDI = date(2026, 11, 2)  # un lundi


@pytest.fixture
def reg(tmp_path):
    return pt.Registre(tmp_path / "test.db")


def at(jour: date, hhmm: str) -> datetime:
    return datetime.combine(jour, datetime.strptime(hhmm, "%H:%M").time())


def test_employes(reg):
    e = reg.ajouter_employe("Awa Diallo", "E001", "Compta", "08:00", "17:00", quand=at(LUNDI, "07:00"))
    assert reg.employe("E001").nom == "Awa Diallo"
    with pytest.raises(ValueError):
        reg.ajouter_employe("Awa Diallo")
    with pytest.raises(ValueError):
        reg.ajouter_employe("Autre", "E001")
    e = reg.modifier_employe("Awa Diallo", service="RH", heure_debut="09:00")
    assert (e.service, e.heure_debut) == ("RH", "09:00")
    reg.modifier_employe("Awa Diallo", actif=0)
    assert reg.employes(actifs=True) == []


def test_journee_retard_depart(reg):
    e = reg.ajouter_employe("Awa", debut="08:00", fin="17:00", quand=at(LUNDI, "07:00"))
    reg.pointer(e, at(LUNDI, "08:20"))
    reg.pointer(e, at(LUNDI, "12:30"))
    reg.pointer(e, at(LUNDI, "16:30"))
    j = reg.journee(e, LUNDI)
    assert j.arrivee == at(LUNDI, "08:20") and j.depart == at(LUNDI, "16:30")
    assert j.retard == 20 and j.depart_anticipe == 30
    assert j.duree == 8 * 60 + 10
    assert j.statut == "Présent"


def test_tolerance(reg):
    e = reg.ajouter_employe("Awa", quand=at(LUNDI, "07:00"))
    reg.pointer(e, at(LUNDI, "08:04"))
    j = reg.journee(e, LUNDI)
    assert j.retard == 0 and j.depart is None and j.statut.startswith("Présent")


def test_annulation(reg):
    e = reg.ajouter_employe("Awa", quand=at(LUNDI, "07:00"))
    pid = reg.pointer(e, at(LUNDI, "07:55"))
    reg.pointer(e, at(LUNDI, "09:10"))
    assert reg.annuler(pid, "doublon")
    assert not reg.annuler(pid, "deux fois")
    assert reg.journee(e, LUNDI).arrivee == at(LUNDI, "09:10")
    assert len(reg.historique(e, LUNDI, LUNDI)) == 2  # l'annulé reste visible


def test_type_prochain_passage(reg):
    e = reg.ajouter_employe("Awa", quand=at(LUNDI, "07:00"))
    assert reg.type_prochain_passage(e, at(LUNDI, "08:00"))[0] == "arrivee"
    reg.pointer(e, at(LUNDI, "08:00"))
    assert reg.type_prochain_passage(e, at(LUNDI, "08:01"))[0] == "doublon"
    assert reg.type_prochain_passage(e, at(LUNDI, "17:00"))[0] == "depart"
    assert reg.type_prochain_passage(e, at(LUNDI + timedelta(days=1), "08:00"))[0] == "arrivee"


def test_rapport_absences_weekend_embauche(reg):
    a = reg.ajouter_employe("Awa", quand=at(LUNDI, "07:00"))
    b = reg.ajouter_employe("Moussa", quand=at(LUNDI + timedelta(days=2), "07:00"))  # embauché mercredi
    for i in range(5):  # Awa vient du lundi au vendredi, en retard le mardi
        jour = LUNDI + timedelta(days=i)
        reg.pointer(a, at(jour, "08:30" if i == 1 else "07:58"))
        reg.pointer(a, at(jour, "17:00"))
    reg.pointer(b, at(LUNDI + timedelta(days=3), "08:00"))  # Moussa vient seulement le jeudi
    lignes, resume = reg.rapport(LUNDI, LUNDI + timedelta(days=6), {1, 2, 3, 4, 5},
                                 aujourd_hui=LUNDI + timedelta(days=10))
    stats = {e.nom: (s, minutes) for e, s, minutes in resume}
    sa, minutes_a = stats["Awa"]
    assert (sa["presents"], sa["absences"], sa["retards"], sa["minutes_retard"]) == (5, 0, 1, 30)
    assert minutes_a == pytest.approx(4 * (9 * 60 + 2) + (8 * 60 + 30))
    sb, _ = stats["Moussa"]
    assert (sb["ouvres"], sb["presents"], sb["absences"]) == (3, 1, 2)  # mer, jeu, ven ; pas avant l'embauche
    assert all(j.jour.isoweekday() <= 5 for _, j in lignes)  # pas de ligne le week-end


def test_rapport_pas_absent_le_jour_meme(reg):
    reg.ajouter_employe("Awa", quand=at(LUNDI, "07:00"))
    _, resume = reg.rapport(LUNDI, LUNDI, {1, 2, 3, 4, 5}, aujourd_hui=LUNDI)
    assert resume[0][1]["absences"] == 0


def test_validateur_confirme_apres_plusieurs_images():
    v = pt.Validateur(images=5, seuil_vivant=0.8)
    for i in range(4):
        assert v.observer(1, "Awa", 0.6, 0.95, i * 0.1) is None
    d = v.observer(1, "Awa", 0.6, 0.95, 0.5)
    assert d.type == "reconnu" and d.nom == "Awa"
    assert v.observer(1, "Awa", 0.6, 0.95, 10) is None  # déjà traité : pas de double pointage


def test_validateur_tolere_une_image_ratee():
    v = pt.Validateur(images=5, seuil_vivant=None)
    noms = ["Awa", "Awa", "Inconnu", "Awa", "Awa", "Awa"]
    decisions = [v.observer(1, n, 0.5, None, i) for i, n in enumerate(noms)]
    assert decisions[-1].type == "reconnu"


def test_validateur_fraude_puis_reessai():
    v = pt.Validateur(images=3, seuil_vivant=0.8, reessai=3)
    for i in range(2):
        v.observer(1, "Awa", 0.6, 0.2, i)
    d = v.observer(1, "Awa", 0.6, 0.2, 2)
    assert d.type == "fraude" and d.vivant == pytest.approx(0.2)
    assert v.observer(1, "Awa", 0.6, 0.99, 3) is None      # pendant le délai
    for t in (6, 7):
        v.observer(1, "Awa", 0.6, 0.99, t)
    assert v.observer(1, "Awa", 0.6, 0.99, 8).type == "reconnu"


def test_validateur_inconnu():
    v = pt.Validateur(images=3, seuil_vivant=0.8)
    for i in range(2):
        v.observer(1, "Inconnu", 0.1, 0.9, i)
    assert v.observer(1, "Inconnu", 0.1, 0.9, 2).type == "inconnu"


def test_arguments():
    assert pt.heure_valide("8h05") == "08:05"
    with pytest.raises(argparse.ArgumentTypeError):
        pt.heure_valide("25:00")
    assert pt.jours_ouvres("1-5") == {1, 2, 3, 4, 5}
    assert pt.jours_ouvres("1-3,6") == {1, 2, 3, 6}
    with pytest.raises(argparse.ArgumentTypeError):
        pt.jours_ouvres("0-9")
    assert pt.mois_arg("2026-11") == (2026, 11)
    assert pt.datetime_arg("07/11/2026 08:05") == datetime(2026, 11, 7, 8, 5)
    assert pt.duree_texte(125) == "2h05"


def test_cli_corriger_et_rapport_csv(tmp_path, capsys):
    base = str(tmp_path / "cli.db")
    reg = pt.Registre(base)
    reg.ajouter_employe("Awa Diallo", "E001", quand=at(LUNDI, "07:00"))
    pt.main(["--base", base, "corriger", "Awa Diallo", "--le", "2026-11-02 08:15", "--motif", "panne"])
    pt.main(["--base", base, "corriger", "E001", "--le", "2026-11-02 17:05", "--motif", "panne"])
    sortie = tmp_path / "nov.csv"
    pt.main(["--base", base, "rapport", "--du", "2026-11-02", "--au", "2026-11-02", "--csv", str(sortie)])
    contenu = sortie.read_text(encoding="utf-8-sig")
    assert "02/11/2026;E001;Awa Diallo;;08:15;17:05;8h50;15;;Présent" in contenu
    pt.main(["--base", base, "historique", "Awa Diallo", "--jours", "400"])
    assert "ajout manuel : panne" in capsys.readouterr().out
