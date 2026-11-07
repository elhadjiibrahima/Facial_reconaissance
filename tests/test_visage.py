"""Tests de la logique (sans webcam ni modèles)."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import visage as vz  # noqa: E402


def unitaire(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_softmax_somme_a_un():
    p = vz.softmax(np.array([1.0, 2.0, 3.0]))
    assert p.sum() == pytest.approx(1.0)
    assert np.argmax(p) == 2


def test_iou():
    assert vz.iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert vz.iou((0, 0, 10, 10), (20, 20, 10, 10)) == 0.0
    assert vz.iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(50 / 150)


def test_identifier():
    connus = {
        "Awa": np.stack([unitaire([1, 0, 0]), unitaire([0.9, 0.1, 0])]),
        "Moussa": np.stack([unitaire([0, 1, 0])]),
    }
    assert vz.identifier(unitaire([1, 0.05, 0]), connus)[0] == "Awa"
    assert vz.identifier(unitaire([0.05, 1, 0]), connus)[0] == "Moussa"
    assert vz.identifier(unitaire([0, 0, 1]), connus)[0] == "Inconnu"
    assert vz.identifier(unitaire([1, 0, 0]), {})[0] == "Inconnu"


def test_nom_valide():
    assert vz.nom_valide(" Awa ") == "Awa"
    for mauvais in ["", "../x", "a/b", ".cache", "a:b"]:
        with pytest.raises(argparse.ArgumentTypeError):
            vz.nom_valide(mauvais)


def test_stockage(tmp_path, monkeypatch):
    monkeypatch.setattr(vz, "VISAGES", tmp_path / "visages")
    assert vz.charger_visages() == {}
    assert vz.sauver_empreintes("Awa", [unitaire([1, 0]), unitaire([0, 1])]) == 2
    assert vz.sauver_empreintes("Awa", [unitaire([1, 1])]) == 3
    assert vz.charger_visages()["Awa"].shape == (3, 2)


def _visage(boite, nom, emo):
    probas = np.full(len(vz.EMOTIONS), 0.01)
    probas[vz.EMOTIONS.index(emo)] = 1.0
    return vz.Visage(boite, nom, 0.5, probas / probas.sum())


def test_suivi_lisse_emotion_et_garde_le_nom():
    s = vz.Suivi(lissage=0.6)
    for t in range(5):
        r = s.mettre_a_jour([_visage((100, 100, 80, 80), "Awa", "Joie")], maintenant=t * 0.1)
    piste = r[0].piste
    # une image isolée « Colere » ne doit pas faire basculer l'émotion affichée
    r = s.mettre_a_jour([_visage((102, 101, 80, 80), "Inconnu", "Colere")], maintenant=0.6)
    assert r[0].piste == piste
    assert r[0].emotion == "Joie"
    assert r[0].nom == "Awa"


def test_suivi_deux_personnes_et_oubli():
    s = vz.Suivi(oubli=1.0)
    r = s.mettre_a_jour([_visage((0, 0, 50, 50), "A", "Neutre"),
                         _visage((300, 0, 50, 50), "B", "Joie")], maintenant=0)
    assert r[0].piste != r[1].piste
    s.mettre_a_jour([], maintenant=5)
    assert s.pistes == {}


def test_rapport(tmp_path, capsys):
    j = tmp_path / "humeurs.csv"
    j.write_text("date,nom,emotion,confiance\n"
                 "2026-01-01T10:00:00,Awa,Joie,0.9\n"
                 "2026-01-01T10:00:05,Awa,Joie,0.8\n"
                 "2026-01-01T10:00:10,Awa,Neutre,0.7\n", encoding="utf-8")
    resume = vz.resumer_journal(str(j))
    assert resume["Awa"]["Joie"] == 2
    vz.main(["rapport", str(j)])
    assert "humeur dominante : Joie" in capsys.readouterr().out


def test_parseur():
    p = vz.construire_parseur()
    a = p.parse_args(["lancer", "--video", "x.mp4", "--sauter", "2"])
    assert a.video == "x.mp4" and a.sauter == 2
