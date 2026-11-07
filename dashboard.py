#!/usr/bin/env python3
"""
Petit dashboard web (stdlib uniquement) pour consulter les présences.

Usage :
  python dashboard.py
  puis ouvrir http://127.0.0.1:8080
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import pointage as pt
from logging_config import log

PORT = 8080

class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        log.info("%s - %s", self.address_string(), format % args)

    def _send(self, code: int, body: bytes, content_type: str = "text/html; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            self._send(200, self._page_principale().encode())
        elif parsed.path == "/api/aujourdhui":
            self._send(200, json.dumps(self._donnees_aujourdhui(), ensure_ascii=False).encode(),
                       "application/json")
        elif parsed.path == "/api/rapport":
            qs = parse_qs(parsed.query)
            mois = qs.get("mois", [None])[0]
            self._send(200, json.dumps(self._donnees_rapport(mois), ensure_ascii=False).encode(),
                       "application/json")
        else:
            self._send(404, b"Not found")

    def _donnees_aujourdhui(self):
        reg = pt.Registre()
        aujourd_hui = date.today()
        lignes = []
        for e in reg.employes(actifs=True):
            j = reg.journee(e, aujourd_hui)
            lignes.append({
                "nom": e.nom,
                "matricule": e.matricule or "",
                "service": e.service,
                "statut": j.statut,
                "arrivee": j.arrivee.strftime("%H:%M") if j.arrivee else "",
                "depart": j.depart.strftime("%H:%M") if j.depart else "",
                "retard": j.retard,
            })
        return {"date": aujourd_hui.isoformat(), "employes": lignes}

    def _donnees_rapport(self, mois_str: str | None):
        reg = pt.Registre()
        if mois_str and len(mois_str) == 7:
            an, mo = map(int, mois_str.split("-"))
            debut = date(an, mo, 1)
            if mo == 12:
                fin = date(an + 1, 1, 1) - timedelta(days=1)
            else:
                fin = date(an, mo + 1, 1) - timedelta(days=1)
        else:
            aujourd = date.today()
            debut = aujourd.replace(day=1)
            fin = aujourd

        _, resume = reg.rapport(debut, fin, {1, 2, 3, 4, 5})
        resultat = []
        for e, stats, minutes in resume:
            resultat.append({
                "nom": e.nom,
                "matricule": e.matricule or "",
                "presents": stats.get("presents", 0),
                "absences": stats.get("absences", 0),
                "retards": stats.get("retards", 0),
                "minutes": int(minutes),
            })
        return {"du": debut.isoformat(), "au": fin.isoformat(), "resume": resultat}

    def _page_principale(self) -> str:
        return """<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <title>Dashboard Pointage</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 2rem; background: #f8f9fa; }
    h1 { color: #1a73e8; }
    table { border-collapse: collapse; width: 100%; background: white; margin: 1rem 0; }
    th, td { padding: 0.6rem 1rem; border-bottom: 1px solid #ddd; text-align: left; }
    th { background: #e8f0fe; }
    .ok { color: green; } .warn { color: orange; } .bad { color: red; }
    #status { font-weight: bold; margin-bottom: 1rem; }
  </style>
</head>
<body>
  <h1>Dashboard de pointage</h1>
  <div id="status">Chargement…</div>
  <h2>Aujourd'hui</h2>
  <table id="aujourdhui"><thead><tr>
    <th>Nom</th><th>Matricule</th><th>Service</th><th>Statut</th>
    <th>Arrivée</th><th>Départ</th><th>Retard</th>
  </tr></thead><tbody></tbody></table>

  <h2>Résumé du mois</h2>
  <table id="rapport"><thead><tr>
    <th>Nom</th><th>Matricule</th><th>Présents</th><th>Absences</th><th>Retards</th><th>Heures</th>
  </tr></thead><tbody></tbody></table>

<script>
async function charger() {
  const status = document.getElementById('status');
  try {
    const auj = await fetch('/api/aujourdhui').then(r => r.json());
    const tb = document.querySelector('#aujourdhui tbody');
    tb.innerHTML = auj.employes.map(e => `
      <tr>
        <td>${e.nom}</td><td>${e.matricule}</td><td>${e.service}</td>
        <td class="${e.statut.includes('Absent') ? 'bad' : e.retard ? 'warn' : 'ok'}">${e.statut}</td>
        <td>${e.arrivee}</td><td>${e.depart}</td><td>${e.retard || ''}</td>
      </tr>`).join('');

    const rap = await fetch('/api/rapport').then(r => r.json());
    const tr = document.querySelector('#rapport tbody');
    tr.innerHTML = rap.resume.map(e => `
      <tr>
        <td>${e.nom}</td><td>${e.matricule}</td>
        <td>${e.presents}</td><td>${e.absences}</td><td>${e.retards}</td>
        <td>${Math.floor(e.minutes/60)}h${String(e.minutes%60).padStart(2,'0')}</td>
      </tr>`).join('');

    status.textContent = `Données du ${auj.date} — mis à jour`;
  } catch (err) {
    status.textContent = 'Erreur de chargement : ' + err;
  }
}
charger();
setInterval(charger, 30000);
</script>
</body>
</html>"""

def main():
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    log.info("Dashboard disponible sur http://127.0.0.1:%s", PORT)
    print(f"Dashboard : http://127.0.0.1:{PORT}  (Ctrl+C pour arrêter)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Arrêt du dashboard")
        print("\nArrêt.")

if __name__ == "__main__":
    main()
