// Signs to sentences and sentences to signs, by fixed rules (web/CONVERSATION.md).
// `signs` is always the `signs` array of web/data/signs.json.

// --- signs to a sentence -----------------------------------------------------------

// The Arabic side is Tunisian Arabic in Arabic script, kept simple.
// A NATIVE SPEAKER MUST CHECK EVERY ARABIC STRING BELOW before real use.
const SENTENCES = {
  // group "reason" and "conditions": one sentence per sign
  ENFERMO: { fr: "Je suis malade.", ar: "أنا مريض." },
  CITA: { fr: "J'ai un rendez-vous.", ar: "عندي موعد." },
  FARMACIA: { fr: "J'ai besoin d'un médicament.", ar: "نستحق دواء." },
  URGENTE: { fr: "C'est urgent.", ar: "الحالة مستعجلة." },
  DIABETES: { fr: "Je suis diabétique.", ar: "عندي السكر." },
  ASMA: { fr: "J'ai de l'asthme.", ar: "عندي الربو." },
  CORAZON: { fr: "J'ai une maladie du cœur.", ar: "عندي مرض في القلب." },
  EMBARAZO: { fr: "Je suis enceinte.", ar: "أنا حامل." },
};

// group "everyday": Tunisian Sign Language words, each its own short sentence, kept in
// the place it was signed. Arabic is Tunisian, written in Arabic script: to be checked
// by a native speaker.
Object.assign(SENTENCES, {
  TN_3ASLEMA: { fr: "Bonjour.", ar: "عسلامة." },
  TN_MAR7BA: { fr: "Bienvenue.", ar: "مرحبا." },
  TN_LABES: { fr: "Ça va.", ar: "لاباس." },
  TN_OUI: { fr: "Oui.", ar: "إيه." },
  TN_NON: { fr: "Non.", ar: "لا." },
  TN_T7EB: { fr: "Je veux.", ar: "نحب." },
  TN_N3AWNEK: { fr: "J'ai besoin d'aide.", ar: "نحب معاونة." },
  TN_NEKTEBLK: { fr: "Je vais vous l'écrire.", ar: "نكتبلك." },
  TN_ASSAM: { fr: "Mon nom.", ar: "إسمي." },
  TN_LYOUM: { fr: "Aujourd'hui.", ar: "اليوم." },
  TN_SE7A: { fr: "Ma santé.", ar: "صحتي." },
  TN_SBITAR: { fr: "L'hôpital.", ar: "السبيطار." },
  TN_MOSTAWSAF: { fr: "Le dispensaire.", ar: "المستوصف." },
  TN_DAR: { fr: "La maison.", ar: "الدار." },
  TN_OM: { fr: "Ma mère.", ar: "أمي." },
  TN_BOU: { fr: "Mon père.", ar: "بابا." },
  TN_3AYLA: { fr: "Ma famille.", ar: "عايلتي." },
  TN_TFOL: { fr: "Mon enfant.", ar: "ولدي." },
  TN_5OU: { fr: "Mon frère.", ar: "خويا." },
  TN_O5T: { fr: "Ma sœur.", ar: "أختي." },
  TN_THNIN: { fr: "Lundi.", ar: "الاثنين." },
  TN_THLETH: { fr: "Mardi.", ar: "الثلاثاء." },
  TN_ERB3A: { fr: "Mercredi.", ar: "الأربعاء." },
  TN_5MIS: { fr: "Jeudi.", ar: "الخميس." },
  TN_JOM3A: { fr: "Vendredi.", ar: "الجمعة." },
  TN_SEBT: { fr: "Samedi.", ar: "السبت." },
  TN_A7AD: { fr: "Dimanche.", ar: "الأحد." },
});

// group "symptom": phrases gathered into one sentence, "J'ai ..." / "عندي ..."
const SYMPTOMS = {
  DOLOR: { fr: "mal", ar: "وجيعة" },
  "DOLOR-DE-CABEZA": { fr: "mal à la tête", ar: "وجيعة في راسي" },
  FIEBRE2: { fr: "de la fièvre", ar: "سخانة" },
  TOSER: { fr: "de la toux", ar: "كحة" },
  MAREO: { fr: "des vertiges", ar: "دوخة" },
  VOMITAR: { fr: "des vomissements", ar: "ترجيع" },
  DIARREA: { fr: "la diarrhée", ar: "إسهال" },
  RESPIRAR: { fr: "du mal à respirer", ar: "ضيق في النفس" },
  CANSADO2: { fr: "de la fatigue", ar: "تعب" },
  HERIDA: { fr: "une blessure", ar: "جرح" },
  ALERGIA2: { fr: "une allergie", ar: "حساسية" },
  GRIPE: { fr: "la grippe", ar: "نزلة برد" },
  INSOMNIO: { fr: "des insomnies", ar: "قلة نوم" },
  ANSIEDAD: { fr: "de l'anxiété", ar: "قلق" },
};

const LISTS = {
  fr: { start: "J'ai ", comma: ", ", and: " et " },
  ar: { start: "عندي ", comma: "، ", and: " و" }, // Arabic "and" is written joined to the next word
};

function joinList(phrases, lang) {
  const { comma, and } = LISTS[lang];
  if (phrases.length <= 1) return phrases.join("");
  return phrases.slice(0, -1).join(comma) + and + phrases[phrases.length - 1];
}

function byCode(signs) {
  return new Map((signs || []).map((sign) => [sign.code, sign]));
}

/**
 * The patient's signs, in the order they were made, as sentences in French
 * and Arabic. Unknown codes are ignored and a sign counts once even when
 * repeated. No usable sign gives { fr: "", ar: "" }.
 */
export function signsToSentence(codes, signs) {
  const known = byCode(signs);
  const groups = new Map(); // group id -> codes, in order of first appearance
  const seen = new Set();
  for (const code of codes || []) {
    const sign = known.get(code);
    if (!sign || seen.has(code)) continue;
    seen.add(code);
    // Everyday words (yes, no, hospital, a day...) keep the place where they were signed.
    const key = sign.group === "everyday" ? `everyday:${code}` : sign.group;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(code);
  }
  const out = { fr: [], ar: [] };
  for (const [group, members] of groups) {
    for (const lang of ["fr", "ar"]) {
      const phrases = members.filter((code) => SYMPTOMS[code]).map((code) => SYMPTOMS[code][lang]);
      if (group === "symptom" && phrases.length) {
        out[lang].push(LISTS[lang].start + joinList(phrases, lang) + ".");
      }
      for (const code of members) {
        if (group === "symptom" && SYMPTOMS[code]) continue;
        // A sign without a written sentence (a future sign) says its own name.
        out[lang].push(SENTENCES[code]?.[lang] ?? `${known.get(code)[lang] ?? code}.`);
      }
    }
  }
  return { fr: out.fr.join(" "), ar: out.ar.join(" ") };
}

/** The sign's name in "fr" (default) or "ar"; an unknown code gives the code itself. */
export function signName(code, signs, lang = "fr") {
  const sign = byCode(signs).get(code);
  if (!sign) return code;
  return sign[lang] ?? sign.fr ?? code;
}

// --- a sentence to signs ------------------------------------------------------------

// Words and phrases that call for each sign, besides its French name in signs.json.
const SYNONYMS = {
  ENFERMO: ["malade", "maladie", "malades"],
  CITA: ["rendez-vous"],
  FARMACIA: ["médicament", "médicaments", "pharmacie", "ordonnance", "traitement"],
  URGENTE: ["urgent", "urgente", "urgence", "urgences"],
  DOLOR: ["mal", "douleur", "douleurs", "maux"],
  "DOLOR-DE-CABEZA": ["mal de tête", "mal à la tête", "maux de tête", "migraine", "migraines"],
  FIEBRE2: ["fièvre", "température"],
  TOSER: ["toux", "tousser", "toussez", "tousse"],
  MAREO: ["vertige", "vertiges", "étourdissement", "étourdissements"],
  VOMITAR: ["vomir", "vomissement", "vomissements", "vomi", "vomissez"],
  DIARREA: ["diarrhée"],
  // "du mal à respirer" must not read as "mal" (pain)
  RESPIRAR: ["respirer", "respiration", "souffle", "essoufflé", "essoufflée", "mal à respirer"],
  CANSADO2: ["fatigue", "fatigué", "fatiguée"],
  HERIDA: ["blessure", "blessures", "plaie", "blessé", "blessée"],
  ALERGIA2: ["allergie", "allergies", "allergique"],
  GRIPE: ["grippe", "rhume"],
  INSOMNIO: ["insomnie", "insomnies", "dormir", "dormez", "sommeil", "mal à dormir"],
  ANSIEDAD: ["anxiété", "angoisse", "stress", "stressé", "stressée", "anxieux", "anxieuse"],
  DIABETES: ["diabète", "diabétique"],
  ASMA: ["asthme", "asthmatique"],
  CORAZON: ["cœur", "coeur", "cardiaque"],
  EMBARAZO: ["enceinte", "grossesse"],
  TN_3ASLEMA: ["bonjour", "salut", "salam", "aslema", "bonsoir"],
  TN_MAR7BA: ["bienvenue"],
  TN_LABES: ["ça va", "comment allez-vous", "comment ça va", "vous allez bien"],
  TN_OUI: ["oui", "d'accord"],
  TN_NON: ["non"],
  TN_T7EB: ["voulez", "voulez-vous", "veux", "vouloir", "voudriez", "souhaitez", "aimeriez"],
  TN_N3AWNEK: ["aide", "aider", "aidez", "aiderai", "aiderons", "aidons", "aidera"],
  TN_NEKTEBLK: ["écrire", "écris", "écrit", "écrive", "écrirai", "écrivez", "j'écris"],
  TN_ASSAM: ["nom", "prénom", "appelez", "appelez-vous"],
  TN_LYOUM: ["aujourd'hui"],
  TN_SE7A: ["santé"],
  TN_SBITAR: ["hôpital", "hôpitaux", "l'hôpital"],
  TN_MOSTAWSAF: ["dispensaire", "clinique", "centre de santé"],
  TN_DAR: ["maison", "domicile", "chez vous"],
  TN_OM: ["mère", "maman"],
  TN_BOU: ["père", "papa"],
  TN_3AYLA: ["famille"],
  TN_TFOL: ["enfant", "enfants", "bébé", "fils", "fille"],
  TN_5OU: ["frère"],
  TN_O5T: ["sœur", "soeur"],
};

// What a doctor says in Tunisian Arabic (Derja), typed or dictated, besides the
// sign's Arabic name in signs.json. Spelling does not matter much: vowel marks,
// hamza, ة/ه, "the" (ال، لل، بال...) and a leading "and" (و) are ignored when matching.
// A NATIVE SPEAKER MUST CHECK THESE WORDS before real use.
const SYNONYMS_TN = {
  ENFERMO: ["مريض", "مريضة", "مرضان", "مرضانة", "مرض"],
  CITA: ["موعد", "رنديفو", "رونديفو"],
  FARMACIA: ["دواء", "دوا", "دوايات", "صيدلية", "فرمسية"],
  URGENTE: ["مستعجل", "مستعجلة", "عاجل", "استعجالي", "ورجانس"],
  DOLOR: ["وجيعة", "وجع", "يوجع", "يوجعك", "يوجعني", "توجع", "توجعك", "ألم"],
  "DOLOR-DE-CABEZA": ["وجيعة في راسي", "وجيعة في راسك", "وجيعة في الراس", "يوجعك راسك", "يوجعني راسي",
    "راسك يوجعك", "راسي يوجعني", "شقيقة"],
  FIEBRE2: ["سخانة", "سخون", "سخونية", "حرارة"],
  TOSER: ["كحة", "تكح", "يكح", "سعال"],
  MAREO: ["دوخة", "دايخ", "دايخة", "تدوخ", "يدوخ"],
  VOMITAR: ["ترجيع", "تقلب", "تستفرغ"],
  DIARREA: ["سهال", "كرشك تجري"],
  RESPIRAR: ["ضيق في النفس", "ضيقة نفس", "تتنفس", "التنفس"],
  CANSADO2: ["تعبان", "تعبانة", "عيان", "عيانة", "فشلان", "فشلة"],
  HERIDA: ["مجروح", "مجروحة", "جرحة"],
  ALERGIA2: ["ألرجي"],
  GRIPE: ["نزلة", "مبرد", "مبردة"],
  INSOMNIO: ["ترقد", "ترقدش", "نرقدش", "رقاد", "نوم"],
  ANSIEDAD: ["قلقان", "قلقانة", "متوتر", "متوترة", "خايف", "خايفة", "ستراس"],
  DIABETES: ["السكر", "مريض بالسكر"],
  ASMA: ["الربو"],
  CORAZON: ["القلب", "قلبك"],
  EMBARAZO: ["حامل", "بالحمل"],
  TN_3ASLEMA: ["عسلامة", "عالسلامة", "السلام عليكم", "السلام", "أهلا", "صباح الخير", "مساء الخير"],
  TN_MAR7BA: ["مرحبا بيك", "أهلا وسهلا", "مرحبا بيكم"],
  TN_LABES: ["لاباس", "لا باس", "لباس", "شنحوالك", "شنية أحوالك", "كيفاش حالك", "بخير"],
  TN_OUI: ["إيه", "نعم", "باهي"],
  TN_NON: ["لا", "لالا"],
  TN_T7EB: ["تحب", "نحب", "تحبش", "تحبي", "حاب", "حابب"],
  TN_N3AWNEK: ["نعاونك", "تعاونك", "يعاونك", "نعاونوك", "معاونة", "عاونك", "نساعدك", "مساعدة"],
  TN_NEKTEBLK: ["نكتبلك", "نكتبهولك", "نكتبهالك", "نكتب", "تكتب", "أكتب"],
  TN_ASSAM: ["اسمك", "إسمك", "اسم"],
  TN_LYOUM: ["اليوم", "ليوم"],
  TN_SE7A: ["صحتك", "صحتي", "الصحة"],
  TN_SBITAR: ["سبيطار", "مستشفى"],
  TN_MOSTAWSAF: ["مستوصف", "مصحة", "كلينيك"],
  TN_DAR: ["دارك", "حوش", "الحوش"],
  TN_OM: ["أمك", "أمي", "يما", "والدتك"],
  TN_BOU: ["بوك", "بابا", "باباك", "والدك"],
  TN_3AYLA: ["عايلتك", "عايلتي", "عائلة", "عائلتك"],
  TN_TFOL: ["طفلة", "ولد", "ولدك", "بنت", "بنتك", "صغار", "ولادك"],
  TN_5OU: ["خوك", "خويا"],
  TN_O5T: ["أختك", "أختي"],
  TN_THNIN: ["نهار الاثنين", "يوم الاثنين", "الإثنين"],
  TN_THLETH: ["نهار الثلاثاء", "يوم الثلاثاء"],
  TN_ERB3A: ["نهار الأربعاء", "يوم الأربعاء", "الاربعاء"],
  TN_5MIS: ["نهار الخميس", "يوم الخميس"],
  TN_JOM3A: ["نهار الجمعة", "يوم الجمعة"],
  TN_SEBT: ["نهار السبت", "يوم السبت"],
  TN_A7AD: ["نهار الأحد", "يوم الأحد"],
};

const ARABIC_LETTER = /[ء-ي]/;

/** True when `text` has Arabic letters (the doctor wrote or dictated in Tounsi). */
export function isArabic(text) {
  return ARABIC_LETTER.test(String(text ?? ""));
}

// "the", alone or after a preposition: و+ال, ب+ال, ف+ال, ك+ال, ل+ال (written لل), ال.
const ARTICLES = ["وال", "بال", "فال", "كال", "لل", "ال"];
// Words whose meaning changes without "the": اليوم today / يوم a day, الاثنين Monday / اثنين two,
// الجمعة Friday / جمعة a week, الأحد Sunday / أحد anyone.
const KEEP_ARTICLE = new Set(["يوم", "اثنين", "جمعه", "احد"]);

/** An Arabic word without "the" (so السبيطار, للسبيطار and سبيطار are one word). */
function withoutArticle(token) {
  if (!ARABIC_LETTER.test(token)) return token;
  for (const prefix of ARTICLES) {
    if (token.startsWith(prefix) && token.length - prefix.length >= 2) {
      const bare = token.slice(prefix.length);
      return KEEP_ARTICLE.has(bare) ? `ال${bare}` : bare;
    }
  }
  return token;
}

/**
 * Lower case, no accents, "œ" as "oe"; in Arabic no vowel marks, no hamza seat,
 * no stretching, ة as ه and ى as ي; then the words.
 */
function rawWords(text) {
  return String(text ?? "")
    .toLowerCase()
    .replace(/œ/g, "oe")
    .replace(/æ/g, "ae")
    .normalize("NFD")                          // أ إ آ ؤ ئ become the bare letter + a mark
    .replace(/[̀-ͯ]/g, "")
    .replace(/[ً-ٰٟـ]/g, "")
    .replace(/ة/g, "ه")
    .replace(/ى/g, "ي")
    .split(/[^a-z0-9ء-ي]+/)
    .filter(Boolean);
}

function words(text) {
  return rawWords(text).map(withoutArticle);
}

/** The ways one word of the doctor's sentence can be read: as written, and without a leading و ("and"). */
function readings(token) {
  const out = [withoutArticle(token)];
  if (token.length > 2 && token.startsWith("و")) out.push(withoutArticle(token.slice(1)));
  return out;
}

const tables = new WeakMap();

/** First word -> [{ words, code }], longest phrase first. Built once per signs array. */
function phraseTable(signs) {
  if (tables.has(signs)) return tables.get(signs);
  const table = new Map();
  const add = (phrase, code) => {
    const tokens = words(phrase);
    if (!tokens.length) return;
    const list = table.get(tokens[0]) ?? [];
    if (list.some((entry) => entry.words.join(" ") === tokens.join(" "))) return; // first one wins
    list.push({ words: tokens, code });
    table.set(tokens[0], list);
  };
  for (const sign of signs || []) {
    add(sign.fr, sign.code);
    for (const phrase of SYNONYMS[sign.code] ?? []) add(phrase, sign.code);
  }
  for (const sign of signs || []) {
    if (sign.ar) add(sign.ar, sign.code);
    for (const phrase of SYNONYMS_TN[sign.code] ?? []) add(phrase, sign.code);
  }
  for (const list of table.values()) list.sort((a, b) => b.words.length - a.words.length);
  if (signs) tables.set(signs, table);
  return table;
}

/**
 * The signs to show the patient for a doctor's sentence, in French or Tunisian
 * Arabic: whole words, case, accents and Arabic spelling variants ignored, the
 * longest phrase first ("mal à la tête" before "mal"), each sign once, in order
 * of appearance.
 */
export function textToSigns(text, signs) {
  const table = phraseTable(signs);
  const tokens = rawWords(text).map(readings);
  const found = [];
  let i = 0;
  while (i < tokens.length) {
    let match = null;
    for (const first of tokens[i]) {
      match = (table.get(first) ?? []).find((entry) =>
        entry.words.every((word, k) => (k === 0 ? word === first : tokens[i + k]?.includes(word))),
      );
      if (match) break;
    }
    if (match) {
      if (!found.includes(match.code)) found.push(match.code);
      i += match.words.length;
    } else {
      i += 1;
    }
  }
  return found;
}

/**
 * The words in a doctor's sentence that show this sign to the patient: in French
 * ("fr", default) the sign's French name first, then its synonyms; in Tounsi ("tn")
 * its Arabic name, then the Tunisian words. An unknown code gives [].
 */
export function signWords(code, signs, lang = "fr") {
  const sign = byCode(signs).get(code);
  if (!sign) return [];
  const seen = new Set();
  const list = lang === "tn" ? [sign.ar, ...(SYNONYMS_TN[code] ?? [])] : [sign.fr, ...(SYNONYMS[code] ?? [])];
  return list.filter((word) => {
    // "l'hôpital" and "j'écris" add nothing to "hôpital" and "écris": elided letters are ignored.
    const key = words(word).filter((token) => token.length > 1).join(" ");
    if (!key || seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}
