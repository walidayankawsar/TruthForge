"""
Sample data generator for TruthForge (pipeline testing ONLY, not for thesis results).
Usage: python make_sample_data.py          -> writes truthforge_data.json (600 samples)
       python make_sample_data.py 3000     -> custom size
"""
import json
import random
import sys

random.seed(42)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 600

TOPICS = {
    "bn": ["নির্বাচন", "বন্যা পরিস্থিতি", "জ্বালানির দাম", "শিক্ষা বাজেট", "টিকা কর্মসূচি",
           "চালের বাজার", "বিদ্যুৎ সংকট", "যানজট নিরসন", "ডিজিটাল সেবা", "স্বাস্থ্যসেবা"],
    "hi": ["चुनाव", "बाढ़ की स्थिति", "ईंधन की कीमतें", "शिक्षा बजट", "टीकाकरण अभियान",
           "अनाज का बाज़ार", "बिजली संकट", "ट्रैफिक सुधार", "डिजिटल सेवाएँ", "स्वास्थ्य सेवा"],
    "en": ["the election", "flood relief", "fuel prices", "the education budget", "the vaccination drive",
           "the food market", "the power shortage", "traffic reforms", "digital services", "public healthcare"],
}

FAKE = {
    "bn": ["সূত্র জানিয়েছে, {t} নিয়ে সরকার সত্য গোপন করছে। সবাইকে শেয়ার করুন!",
           "ভয়ংকর খবর! {t} নিয়ে বিরোধী দল গোপনে বিদেশ থেকে টাকা নিয়েছে, প্রমাণ ফাঁস।",
           "অবিশ্বাস্য! {t} নিয়ে এই উপায়ে এক সপ্তাহেই সব সমস্যা শেষ, ডাক্তাররা চান না আপনি জানুন।",
           "{t} নিয়ে যা বলা হচ্ছে সবই মিথ্যা, আসল ঘটনা মিডিয়া দেখাচ্ছে না।"],
    "hi": ["सूत्रों के अनुसार {t} पर सरकार सच छिपा रही है। तुरंत सबको शेयर करें!",
           "वायरल दावा: {t} को लेकर विपक्ष को विदेश से गुप्त फंडिंग मिली, सबूत लीक।",
           "डॉक्टर नहीं चाहते कि आप जानें: {t} का यह उपाय एक हफ्ते में सब ठीक कर देगा।",
           "{t} के बारे में जो बताया जा रहा है वह सब झूठ है, मीडिया असली सच नहीं दिखा रहा।"],
    "en": ["Sources say the government is hiding the truth about {t}. Share before it gets deleted!",
           "Shocking leak: opposition secretly received foreign money over {t}, documents exposed.",
           "Doctors don't want you to know: this one trick fixes everything about {t} in a week.",
           "Everything you were told about {t} is a lie, and the media will not show you the real story."],
}

REAL = {
    "bn": ["{t} বিষয়ে সরকারি বিজ্ঞপ্তিতে বলা হয়েছে, নতুন নিয়ম আগামী মাস থেকে কার্যকর হবে।",
           "সংবাদ সম্মেলনে মন্ত্রী {t} সংক্রান্ত পরিসংখ্যান প্রকাশ করেছেন, যা স্বাধীন সংস্থাও যাচাই করেছে।",
           "{t} নিয়ে গবেষণার ফল একটি পিয়ার-রিভিউড জার্নালে প্রকাশিত হয়েছে।",
           "{t} নিয়ে সংসদীয় কমিটি বৈঠক করেছে এবং সুপারিশসহ প্রতিবেদন জমা দিয়েছে।"],
    "hi": ["{t} के बारे में सरकारी अधिसूचना में कहा गया है कि नए नियम अगले महीने से लागू होंगे।",
           "प्रेस कॉन्फ्रेंस में मंत्री ने {t} से जुड़े आंकड़े जारी किए, जिन्हें स्वतंत्र संस्था ने भी सत्यापित किया।",
           "{t} पर शोध के नतीजे एक समीक्षित जर्नल में प्रकाशित हुए हैं।",
           "{t} पर संसदीय समिति ने बैठक की और सिफारिशों के साथ रिपोर्ट सौंपी।"],
    "en": ["An official notice on {t} states that the new rules take effect next month.",
           "At a press conference the minister released figures on {t}, which an independent body also verified.",
           "Research findings on {t} have been published in a peer-reviewed journal.",
           "A parliamentary committee met on {t} and submitted a report with recommendations."],
}

SENSATIONAL = {"bn": ["জরুরি: ", "ব্রেকিং: ", "বিস্ফোরক: "],
               "hi": ["जरूरी: ", "ब्रेकिंग: ", "विस्फोटक: "],
               "en": ["URGENT: ", "BREAKING: ", "EXPLOSIVE: "]}

COUNTRY_BY_LANG = {"bn": ["BD"] * 8 + ["IN"], "hi": ["IN"] * 8 + ["BD"], "en": ["US", "IN", "BD", "UNK"]}
PARTIES = {"IN": ["BJP", "Congress", "AAP", "None"], "BD": ["Awami League", "BNP", "Jatiya Party", "None"],
           "US": ["Democrat", "Republican", "None"], "UNK": ["None"]}
ROLES = ["politician", "journalist", "citizen", "official", "unknown"]


def clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def make_sample(is_fake: bool):
    lang = random.choice(["bn", "hi", "en"])
    topic = random.choice(TOPICS[lang])
    text = random.choice((FAKE if is_fake else REAL)[lang]).format(t=topic)

    # Noise so the task is not trivial: some fakes look neutral, some real items look sensational
    if random.random() < (0.5 if is_fake else 0.1):
        text = random.choice(SENSATIONAL[lang]) + text

    country = random.choice(COUNTRY_BY_LANG[lang])
    meta = {
        "political_party": random.choice(PARTIES[country]),
        "speaker_role": random.choice(ROLES),
        "country": country,
        # Overlapping distributions: credibility is only a WEAK signal
        "source_credibility": round(clip(random.gauss(0.40 if is_fake else 0.60, 0.22)), 2),
        "domain_age": random.randint(1, 8) if (is_fake and random.random() < 0.6) else random.randint(1, 20),
    }
    return {"text": text, "label": int(is_fake), "metadata": meta}


data = [make_sample(i % 2 == 0) for i in range(N)]
random.shuffle(data)

with open("truthforge_data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Wrote {N} samples | fake={sum(d['label'] for d in data)} real={N - sum(d['label'] for d in data)}")
