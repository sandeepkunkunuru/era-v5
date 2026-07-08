"""Fetch the India Wikipedia article in en/hi/te/es as plain text into data/.
Graders re-run the tokenizer, so we commit the exact corpus we trained on for reproducibility."""
import json, os, re, urllib.request, urllib.parse
DATA=os.path.join(os.path.dirname(os.path.abspath(__file__)),"data"); os.makedirs(DATA,exist_ok=True)
PAGES=[("en","en","India"),("hi","hi","भारत"),("te","te","భారతదేశం"),("es","es","India")]

def fetch(sub,title):
    q=urllib.parse.urlencode({"action":"query","prop":"extracts","explaintext":1,
        "redirects":1,"format":"json","titles":title})
    req=urllib.request.Request(f"https://{sub}.wikipedia.org/w/api.php?{q}",
        headers={"User-Agent":"era-v5-assignment/1.0 (educational)"})
    d=json.load(urllib.request.urlopen(req,timeout=60))
    pg=next(iter(d["query"]["pages"].values()))
    return pg.get("extract",""), pg.get("title","")

if __name__=="__main__":
    print(f"{'lang':5}{'title':16}{'chars':>9}{'words':>9}{'uniq':>8}")
    for lang,sub,title in PAGES:
        text,resolved=fetch(sub,title)
        open(f"{DATA}/{lang}.txt","w",encoding="utf-8").write(text)
        w=re.findall(r"\S+",text)
        print(f"{lang:5}{resolved[:15]:16}{len(text):9}{len(w):9}{len(set(w)):8}")
