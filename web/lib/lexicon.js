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
    if (!groups.has(sign.group)) groups.set(sign.group, []);
    groups.get(sign.group).push(code);
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
};

/** Lower case, no accents, "œ" as "oe", then the words. */
function words(text) {
  return String(text ?? "")
    .toLowerCase()
    .replace(/œ/g, "oe")
    .replace(/æ/g, "ae")
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")
    .split(/[^a-z0-9]+/)
    .filter(Boolean);
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
  for (const list of table.values()) list.sort((a, b) => b.words.length - a.words.length);
  if (signs) tables.set(signs, table);
  return table;
}

/**
 * The signs to show the patient for a doctor's sentence: whole words, case and
 * accents ignored, the longest phrase first ("mal à la tête" before "mal"),
 * each sign once, in order of appearance.
 */
export function textToSigns(text, signs) {
  const table = phraseTable(signs);
  const tokens = words(text);
  const found = [];
  let i = 0;
  while (i < tokens.length) {
    const match = (table.get(tokens[i]) ?? []).find((entry) =>
      entry.words.every((word, k) => tokens[i + k] === word),
    );
    if (match) {
      if (!found.includes(match.code)) found.push(match.code);
      i += match.words.length;
    } else {
      i += 1;
    }
  }
  return found;
}
