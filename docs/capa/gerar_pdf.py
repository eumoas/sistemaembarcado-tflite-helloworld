"""Gera o relatorio em PDF a partir do README.md.

Uso (na raiz do repositorio):  python3 docs/capa/gerar_pdf.py
Requer: pacote Python `markdown` e o Google Chrome (impressao headless).
"""

import os
import re
import subprocess
import unicodedata

import markdown

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPO = "https://github.com/eumoas/sistemaembarcado-tflite-helloworld"
SAIDA = os.path.join(RAIZ, "docs", "Relatorio_Atividade4_TFLite_Miriam_Sobral.pdf")
HTML_TMP = os.path.join(RAIZ, "docs", "capa", "relatorio.html")
DATA = "1º de outubro de 2026"


def slug_github(texto, separador="-"):
    """Mesmo identificador que o GitHub cria para cada titulo (mantem acentos)."""
    texto = unicodedata.normalize("NFC", texto).strip().lower()
    texto = re.sub(r"<[^>]+>", "", texto)
    texto = re.sub(r"[^\w\- ]", "", texto)
    return texto.replace(" ", separador)


def corpo_do_readme():
    texto = open(os.path.join(RAIZ, "README.md"), encoding="utf-8").read()
    # A capa do PDF substitui a imagem de abertura e a tabela de identificacao.
    texto = texto[texto.index("\n# ") + 1:]
    html = markdown.markdown(
        texto,
        extensions=["tables", "fenced_code", "toc"],
        extension_configs={"toc": {"slugify": slug_github}},
    )

    def link_absoluto(m):
        attr, alvo = m.group(1), m.group(2)
        if alvo.startswith(("http", "#", "mailto:", "file:")):
            return m.group(0)
        if attr == "src":
            return f'src="file://{os.path.join(RAIZ, alvo)}"'
        tipo = "tree" if alvo.endswith("/") else "blob"
        return f'href="{REPO}/{tipo}/main/{alvo}"'

    return re.sub(r'(href|src)="([^"]+)"', link_absoluto, html)


CAPA = f"""
<section class="capa">
  <div class="capa-topo">
    <p>Pós-graduação em Inteligência Artificial Aplicada</p>
    <p>Turma PG PGIA 2025/2 1</p>
    <p>Unidade Curricular: IA Embarcada e Modelos Compactos — 489780</p>
  </div>
  <div class="capa-meio">
    <p class="capa-atividade">Atividade Avaliativa Prática (4/6)</p>
    <h1 class="capa-titulo">TensorFlow Lite Micro no ESP32 com Wokwi</h1>
    <p class="capa-sub">Hello World e Detector de Ocupação de Sala<br>(DHT22 + LDR, modelo int8 próprio)</p>
    <img src="file://{RAIZ}/docs/imagens/capa.png" alt="">
  </div>
  <div class="capa-base">
    <table>
      <tr><th>Aluna</th><td>Miriam O. A. Sobral</td></tr>
      <tr><th>Professor</th><td>Rodrigo K. Rosa</td></tr>
      <tr><th>Repositório público</th><td><a href="{REPO}">{REPO.replace("https://", "")}</a></td></tr>
      <tr><th>Data</th><td>{DATA}</td></tr>
    </table>
  </div>
</section>
"""

CSS = """
@page {
  size: A4;
  margin: 18mm 17mm 20mm 17mm;
  @bottom-left { content: "Atividade 4/6 · TensorFlow Lite Micro no ESP32 · Miriam O. A. Sobral";
                 font: 8pt "Noto Sans", sans-serif; color: #6b7785; }
  @bottom-right { content: counter(page) " / " counter(pages);
                  font: 8pt "Noto Sans", sans-serif; color: #6b7785; }
}
@page :first { margin: 0; @bottom-left { content: none; } @bottom-right { content: none; } }

* { box-sizing: border-box; }
body { font: 10pt/1.5 "Noto Sans", "DejaVu Sans", sans-serif; color: #1d2733; margin: 0; }
a { color: #1a5fb4; text-decoration: none; }

.capa { height: 297mm; padding: 22mm 20mm 20mm; display: flex; flex-direction: column;
        justify-content: space-between; page-break-after: always;
        border-top: 10mm solid #0f1b2d; }
.capa-topo p { margin: 0; text-align: center; font-size: 11.5pt; line-height: 1.6; }
.capa-topo p:first-child { font-weight: 700; font-size: 13pt; }
.capa-meio { text-align: center; }
.capa-atividade { color: #c0640f; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
                  font-size: 10.5pt; margin: 0 0 4mm; }
.capa-titulo { font-size: 26pt; line-height: 1.15; margin: 0 0 4mm; border: none; color: #0f1b2d; }
.capa-sub { font-size: 13pt; color: #44505e; margin: 0 0 9mm; }
.capa-meio img { width: 100%; border-radius: 3mm; }
.capa-base table { width: 100%; border-collapse: collapse; font-size: 11pt; }
.capa-base th { text-align: left; width: 45mm; color: #44505e; font-weight: 600; }
.capa-base th, .capa-base td { padding: 2.2mm 0; border-bottom: 1px solid #d5dbe2; background: none; }

h1 { font-size: 19pt; color: #0f1b2d; margin: 0 0 4mm; line-height: 1.2; }
h2 { font-size: 14.5pt; color: #0f1b2d; border-bottom: 2px solid #ff9e3d; padding-bottom: 1.5mm;
     margin: 9mm 0 3mm; page-break-after: avoid; }
h3 { font-size: 12pt; color: #16263d; margin: 6mm 0 2mm; page-break-after: avoid; }
h4 { font-size: 10.5pt; margin: 5mm 0 2mm; page-break-after: avoid; }
h2 + *, h3 + *, h4 + * { page-break-before: avoid; }
p, li { orphans: 3; widows: 3; }
ul, ol { padding-left: 6mm; }

table { border-collapse: collapse; width: 100%; margin: 3mm 0 4mm; font-size: 8.6pt;
        page-break-inside: auto; }
tr { page-break-inside: avoid; }
th, td { border: 1px solid #d5dbe2; padding: 1.3mm 2mm; vertical-align: top; text-align: left; }
th { background: #eef2f7; font-weight: 700; }

code { font: 8.6pt "DejaVu Sans Mono", monospace; background: #f1f4f8; padding: 0 1mm; border-radius: 1mm; }
pre { background: #f5f7fa; border: 1px solid #dde3ea; border-left: 3px solid #43c59e; border-radius: 1.5mm;
      padding: 3mm; font-size: 8.2pt; line-height: 1.4; white-space: pre-wrap; word-break: break-word;
      page-break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 8.2pt; }
blockquote { margin: 3mm 0; padding: 2mm 4mm; background: #fff6ec; border-left: 3px solid #ff9e3d; }
blockquote p { margin: 0; }

img { max-width: 100%; }
p[align="center"] { text-align: center; page-break-inside: avoid; margin: 3mm 0 1mm; }
/* imagem e legenda ficam sempre na mesma pagina */
p[align="center"]:has(img) { break-after: avoid; page-break-after: avoid; }
p[align="center"] img { max-height: 100mm; object-fit: contain; border: 1px solid #d5dbe2; }
p[align="center"] img[width="90%"] { width: auto; max-width: 100%; }
p[align="center"] em { font-size: 8.6pt; color: #44505e; display: block; }
"""


def main():
    html = f"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8">
<title>Relatório Atividade 4/6 — TensorFlow Lite Micro no ESP32 — Miriam O. A. Sobral</title>
<style>{CSS}</style></head>
<body>{CAPA}<main>{corpo_do_readme()}</main></body></html>"""
    with open(HTML_TMP, "w", encoding="utf-8") as f:
        f.write(html)
    subprocess.run([
        "google-chrome", "--headless=new", "--disable-gpu", "--no-sandbox",
        "--no-pdf-header-footer", "--allow-file-access-from-files",
        f"--print-to-pdf={SAIDA}", f"file://{HTML_TMP}",
    ], check=True, stderr=subprocess.DEVNULL)
    os.remove(HTML_TMP)
    print("PDF gerado:", SAIDA)


if __name__ == "__main__":
    main()
