"""Normalize Turkish display labels for AUTSL and Android assets."""

from __future__ import annotations

import csv
import os
from pathlib import Path


ROOT = Path(__file__).parent
CSV_PATH = ROOT / "AUTSL" / "SignList_ClassId_TR_EN.csv"
CORRECTED_CSV_PATH = ROOT / "AUTSL" / "SignList_ClassId_TR_EN_corrected.csv"
LABELS_TR_PATH = ROOT / "android" / "app" / "src" / "main" / "assets" / "labels_tr.txt"

CORRECTIONS = {
    "abla": "Abla",
    "acele": "Acele",
    "acikmak": "Acıkmak",
    "afiyet_olsun": "Afiyet olsun",
    "agabey": "Ağabey",
    "agac": "Ağaç",
    "agir": "Ağır",
    "aglamak": "Ağlamak",
    "aile": "Aile",
    "akilli": "Akıllı",
    "akilsiz": "Akılsız",
    "akraba": "Akraba",
    "alisveris": "Alışveriş",
    "anahtar": "Anahtar",
    "anne": "Anne",
    "arkadas": "Arkadaş",
    "ataturk": "Atatürk",
    "ayakkabi": "Ayakkabı",
    "ayna": "Ayna",
    "ayni": "Aynı",
    "baba": "Baba",
    "bahce": "Bahçe",
    "bakmak": "Bakmak",
    "bal": "Bal",
    "bardak": "Bardak",
    "bayrak": "Bayrak",
    "bayram": "Bayram",
    "bebek": "Bebek",
    "bekar": "Bekar",
    "beklemek": "Beklemek",
    "ben": "Ben",
    "benzin": "Benzin",
    "beraber": "Beraber",
    "bilgi_vermek": "Bilgi vermek",
    "biz": "Biz",
    "calismak": "Çalışmak",
    "carsamba": "Çarşamba",
    "catal": "Çatal",
    "cay": "Çay",
    "caydanlik": "Çaydanlık",
    "cekic": "Çekiç",
    "cirkin": "Çirkin",
    "cocuk": "Çocuk",
    "corba": "Çorba",
    "cuma": "Cuma",
    "cumartesi": "Cumartesi",
    "cuzdan": "Cüzdan",
    "dakika": "Dakika",
    "dede": "Dede",
    "degistirmek": "Değiştirmek",
    "devirmek": "Devirmek",
    "devlet": "Devlet",
    "doktor": "Doktor",
    "dolu": "Dolu",
    "dugun": "Düğün",
    "dun": "Dün",
    "dusman": "Düşman",
    "duvar": "Duvar",
    "eczane": "Eczane",
    "eldiven": "Eldiven",
    "emek": "Emek",
    "emekli": "Emekli",
    "erkek": "Erkek",
    "et": "Et",
    "ev": "Ev",
    "evet": "Evet",
    "evli": "Evli",
    "ezberlemek": "Ezberlemek",
    "fil": "Fil",
    "fotograf": "Fotoğraf",
    "futbol": "Futbol",
    "gecmis": "Geçmiş",
    "gecmis_olsun": "Geçmiş olsun",
    "getirmek": "Getirmek",
    "gol": "Göl",
    "gomlek": "Gömlek",
    "gormek": "Görmek",
    "gostermek": "Göstermek",
    "gulmek": "Gülmek",
    "hafif": "Hafif",
    "hakli": "Haklı",
    "hali": "Halı",
    "hasta": "Hasta",
    "hastane": "Hastane",
    "hata": "Hata",
    "havlu": "Havlu",
    "hayir": "Hayır",
    "hayirli_olsun": "Hayırlı olsun",
    "hayvan": "Hayvan",
    "hediye": "Hediye",
    "helal": "Helal",
    "hep": "Hep",
    "hic": "Hiç",
    "hoscakal": "Hoşça kal",
    "icmek": "İçmek",
    "igne": "İğne",
    "ilac": "İlaç",
    "ilgilenmemek": "İlgilenmemek",
    "isik": "Işık",
    "itmek": "İtmek",
    "iyi": "İyi",
    "kacmak": "Kaçmak",
    "kahvalti": "Kahvaltı",
    "kalem": "Kalem",
    "kalorifer": "Kalorifer",
    "kapi": "Kapı",
    "kardes": "Kardeş",
    "kavsak": "Kavşak",
    "kaza": "Kaza",
    "kemer": "Kemer",
    "keske": "Keşke",
    "kim": "Kim",
    "kimlik": "Kimlik",
    "kira": "Kira",
    "kitap": "Kitap",
    "kiyma": "Kıyma",
    "kiz": "Kız",
    "koku": "Koku",
    "kolonya": "Kolonya",
    "komur": "Kömür",
    "kopek": "Köpek",
    "kopru": "Köprü",
    "kotu": "Kötü",
    "kucak": "Kucak",
    "leke": "Leke",
    "maas": "Maaş",
    "makas": "Makas",
    "masa": "Masa",
    "masallah": "Maşallah",
    "melek": "Melek",
    "memnun_olmak": "Memnun olmak",
    "mendil": "Mendil",
    "merdiven": "Merdiven",
    "misafir": "Misafir",
    "mudur": "Müdür",
    "musluk": "Musluk",
    "nasil": "Nasıl",
    "neden": "Neden",
    "nerede": "Nerede",
    "nine": "Nine",
    "ocak": "Ocak",
    "oda": "Oda",
    "odun": "Odun",
    "ogretmen": "Öğretmen",
    "okul": "Okul",
    "olimpiyat": "Olimpiyat",
    "olmaz": "Olmaz",
    "olur": "Olur",
    "onlar": "Onlar",
    "orman": "Orman",
    "oruc": "Oruç",
    "ozur_dilemek": "Özür dilemek",
    "pamuk": "Pamuk",
    "pantolon": "Pantolon",
    "para": "Para",
    "pastirma": "Pastırma",
    "patates": "Patates",
    "pazar": "Pazar",
    "pazartesi": "Pazartesi",
    "pencere": "Pencere",
    "persembe": "Perşembe",
    "piknik": "Piknik",
    "polis": "Polis",
    "psikoloji": "Psikoloji",
    "rica_etmek": "Rica etmek",
    "saat": "Saat",
    "sabun": "Sabun",
    "salca": "Salça",
    "sali": "Salı",
    "sampiyon": "Şampiyon",
    "sapka": "Şapka",
    "savas": "Savaş",
    "seker": "Şeker",
    "selam": "Selam",
    "semsiye": "Şemsiye",
    "sen": "Sen",
    "senet": "Senet",
    "serbest": "Serbest",
    "ses": "Ses",
    "sevmek": "Sevmek",
    "seytan": "Şeytan",
    "sinir": "Sınır",
    "siz": "Siz",
    "soylemek": "Söylemek",
    "soz": "Söz",
    "sut": "Süt",
    "tamam": "Tamam",
    "tarak": "Tarak",
    "tarih": "Tarih",
    "tatil": "Tatil",
    "tatli": "Tatlı",
    "tavan": "Tavan",
    "tehlike": "Tehlike",
    "telefon": "Telefon",
    "terazi": "Terazi",
    "terzi": "Terzi",
    "tesekkur": "Teşekkür",
    "tornavida": "Tornavida",
    "turkiye": "Türkiye",
    "turuncu": "Turuncu",
    "tuvalet": "Tuvalet",
    "un": "Un",
    "uzak": "Uzak",
    "uzgun": "Üzgün",
    "var": "Var",
    "vergi": "Vergi",
    "yakin": "Yakın",
    "yalniz": "Yalnız",
    "yanlis": "Yanlış",
    "yapmak": "Yapmak",
    "yarabandi": "Yara bandı",
    "yardim": "Yardım",
    "yarin": "Yarın",
    "yasak": "Yasak",
    "yastik": "Yastık",
    "yatak": "Yatak",
    "yavas": "Yavaş",
    "yemek": "Yemek",
    "yemek_pisirmek": "Yemek pişirmek",
    "yildiz": "Yıldız",
    "yok": "Yok",
    "yol": "Yol",
    "yorgun": "Yorgun",
    "yumurta": "Yumurta",
    "zaman": "Zaman",
    "zor": "Zor",
}


def main():
    with CSV_PATH.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    missing = [row["TR"] for row in rows if row["TR"] not in CORRECTIONS]
    if missing:
        raise ValueError(f"Missing corrections for: {missing}")

    for row in rows:
        row["TR"] = CORRECTIONS[row["TR"]]

    temp_csv_path = CSV_PATH.with_suffix(".csv.tmp")
    with temp_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ClassId", "TR", "EN"])
        writer.writeheader()
        writer.writerows(rows)

    target_csv_path = CSV_PATH
    try:
        os.replace(temp_csv_path, CSV_PATH)
    except PermissionError:
        target_csv_path = CORRECTED_CSV_PATH
        with CORRECTED_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["ClassId", "TR", "EN"])
            writer.writeheader()
            writer.writerows(rows)
        temp_csv_path.unlink(missing_ok=True)

    labels = [row["TR"] for row in sorted(rows, key=lambda item: int(item["ClassId"]))]
    LABELS_TR_PATH.write_text("\n".join(labels) + "\n", encoding="utf-8")

    print(f"Updated: {target_csv_path}")
    print(f"Updated: {LABELS_TR_PATH}")
    print(f"Total labels: {len(labels)}")


if __name__ == "__main__":
    main()
