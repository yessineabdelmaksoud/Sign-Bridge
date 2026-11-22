import assert from "node:assert/strict";
import test from "node:test";

import * as LEXICON from "../lib/lexicon.js";
import { signName, signsToSentence, textToSigns } from "../lib/lexicon.js";
import { readJson } from "./helpers.js";

const { signs } = readJson("data/signs.json");
const fr = (codes) => signsToSentence(codes, signs).fr;
const find = (text) => textToSigns(text, signs);

// --- signsToSentence -----------------------------------------------------------

test("the example sentence of the contract", () => {
  assert.equal(fr(["FIEBRE2", "TOSER", "DOLOR-DE-CABEZA"]), "J'ai de la fièvre, de la toux et mal à la tête.");
});

test("the example message of the contract", () => {
  assert.equal(fr(["ENFERMO", "FIEBRE2"]), "Je suis malade. J'ai de la fièvre.");
});

test("one and two symptoms", () => {
  assert.equal(fr(["DOLOR"]), "J'ai mal.");
  assert.equal(fr(["MAREO", "VOMITAR"]), "J'ai des vertiges et des vomissements.");
});

test("every sign has a French and an Arabic sentence", () => {
  for (const sign of signs) {
    const sentence = signsToSentence([sign.code], signs);
    assert.match(sentence.fr, /^\p{Lu}.+\.$/u, sign.code);
    assert.ok(sentence.ar.length > 1, sign.code);
    assert.match(sentence.ar, /[؀-ۿ]/u, sign.code);
  }
});

test("each reason and condition is its own sentence", () => {
  assert.equal(fr(["CITA", "FARMACIA"]), "J'ai un rendez-vous. J'ai besoin d'un médicament.");
  assert.equal(fr(["URGENTE"]), "C'est urgent.");
  assert.equal(fr(["DIABETES", "ASMA", "CORAZON", "EMBARAZO"]),
    "Je suis diabétique. J'ai de l'asthme. J'ai une maladie du cœur. Je suis enceinte.");
});

test("groups keep the order in which they first appear", () => {
  assert.equal(
    fr(["FIEBRE2", "DIABETES", "ENFERMO", "TOSER"]),
    "J'ai de la fièvre et de la toux. Je suis diabétique. Je suis malade.",
  );
  assert.equal(
    fr(["EMBARAZO", "URGENTE", "RESPIRAR"]),
    "Je suis enceinte. C'est urgent. J'ai du mal à respirer.",
  );
});

test("a sign repeated in a row counts once", () => {
  assert.equal(fr(["FIEBRE2", "FIEBRE2", "TOSER"]), "J'ai de la fièvre et de la toux.");
  assert.equal(fr(["ENFERMO", "ENFERMO"]), "Je suis malade.");
});

test("a sign repeated later in the same message also counts once", () => {
  assert.equal(fr(["FIEBRE2", "TOSER", "FIEBRE2"]), "J'ai de la fièvre et de la toux.");
});

test("unknown codes are ignored", () => {
  assert.equal(fr(["OTHER", "FIEBRE2", "NOPE", "TOSER"]), "J'ai de la fièvre et de la toux.");
  assert.deepEqual(signsToSentence(["OTHER"], signs), { fr: "", ar: "" });
});

test("no signs gives empty strings", () => {
  assert.deepEqual(signsToSentence([], signs), { fr: "", ar: "" });
  assert.deepEqual(signsToSentence(undefined, signs), { fr: "", ar: "" });
});

test("the Arabic side follows the same rules", () => {
  const { ar } = signsToSentence(["ENFERMO", "FIEBRE2", "TOSER", "DOLOR-DE-CABEZA"], signs);
  assert.equal(ar, "أنا مريض. عندي سخانة، كحة ووجيعة في راسي.");
  assert.equal(signsToSentence(["FIEBRE2", "FIEBRE2"], signs).ar, "عندي سخانة.");
});

// --- signName --------------------------------------------------------------------

test("signName gives the French or Arabic name", () => {
  assert.equal(signName("FIEBRE2", signs, "fr"), "Fièvre");
  assert.equal(signName("FIEBRE2", signs, "ar"), "سخانة");
  assert.equal(signName("TOSER", signs), "Toux");
  assert.equal(signName("NOPE", signs, "fr"), "NOPE");
});

// --- textToSigns -------------------------------------------------------------------

test("every ready-made doctor question finds its sign", () => {
  const questions = {
    "Avez-vous de la fièvre ?": ["FIEBRE2"],
    "Avez-vous mal à la tête ?": ["DOLOR-DE-CABEZA"],
    "Toussez-vous ?": ["TOSER"],
    "Avez-vous des vertiges ?": ["MAREO"],
    "Avez-vous du mal à respirer ?": ["RESPIRAR"],
    "Êtes-vous diabétique ?": ["DIABETES"],
    "Êtes-vous enceinte ?": ["EMBARAZO"],
    "Avez-vous une allergie ?": ["ALERGIA2"],
    "Prenez ce médicament.": ["FARMACIA"],
    "Revenez pour un rendez-vous.": ["CITA"],
  };
  for (const [question, codes] of Object.entries(questions)) {
    assert.deepEqual(find(question), codes, question);
  }
});

test("the longest phrase wins", () => {
  assert.deepEqual(find("J'ai mal de tête"), ["DOLOR-DE-CABEZA"]);
  assert.deepEqual(find("Une maladie du cœur ?"), ["CORAZON"]);
  assert.deepEqual(find("Vous êtes malade ?"), ["ENFERMO"]);
  assert.deepEqual(find("Avez-vous mal ?"), ["DOLOR"]);
});

test("signs come in order of appearance, each once", () => {
  assert.deepEqual(find("De la toux, de la fièvre, encore de la toux ?"), ["TOSER", "FIEBRE2"]);
  assert.deepEqual(find("Douleur ou migraine ? Une douleur forte ?"), ["DOLOR", "DOLOR-DE-CABEZA"]);
  assert.deepEqual(find("mal à la tête et mal au ventre"), ["DOLOR-DE-CABEZA", "DOLOR"]);
});

test("accents and capitals are ignored", () => {
  assert.deepEqual(find("FIEVRE"), ["FIEBRE2"]);
  assert.deepEqual(find("anxiete, ANXIÉTÉ"), ["ANSIEDAD"]);
  assert.deepEqual(find("DIABÈTE"), ["DIABETES"]);
  assert.deepEqual(find("coeur"), ["CORAZON"]);
  assert.deepEqual(find("CŒUR"), ["CORAZON"]);
  assert.deepEqual(find("AVEZ-VOUS MAL A LA TETE ?"), ["DOLOR-DE-CABEZA"]);
});

test("only whole words match", () => {
  assert.deepEqual(find("malheureusement"), []);
  assert.deepEqual(find("toux/fièvre"), ["TOSER", "FIEBRE2"]);
  assert.deepEqual(find("animal"), []);
  assert.deepEqual(find("toussotement"), []);
  assert.deepEqual(find("souffleur"), []);
});

test("apostrophes and punctuation separate words", () => {
  assert.deepEqual(find("l'urgence"), ["URGENTE"]);
  assert.deepEqual(find("d’anxiété"), ["ANSIEDAD"]);
  assert.deepEqual(find("(grippe)"), ["GRIPE"]);
});

test("the synonyms of the contract", () => {
  const expected = {
    ENFERMO: ["malade", "maladie"],
    CITA: ["rendez-vous"],
    FARMACIA: ["médicament", "médicaments", "pharmacie", "ordonnance", "traitement"],
    URGENTE: ["urgent", "urgence"],
    DOLOR: ["mal", "douleur", "douleurs"],
    "DOLOR-DE-CABEZA": ["mal de tête", "mal à la tête", "migraine"],
    FIEBRE2: ["fièvre", "température"],
    TOSER: ["toux", "tousser", "toussez"],
    MAREO: ["vertige", "vertiges", "étourdissement"],
    VOMITAR: ["vomir", "vomissement", "vomissements", "vomi"],
    DIARREA: ["diarrhée"],
    RESPIRAR: ["respirer", "respiration", "souffle", "essoufflé"],
    CANSADO2: ["fatigue", "fatigué", "fatiguée"],
    HERIDA: ["blessure", "plaie", "blessé"],
    ALERGIA2: ["allergie", "allergique"],
    GRIPE: ["grippe", "rhume"],
    INSOMNIO: ["insomnie", "dormir", "sommeil"],
    ANSIEDAD: ["anxiété", "angoisse", "stress", "stressé"],
    DIABETES: ["diabète", "diabétique"],
    ASMA: ["asthme"],
    CORAZON: ["cœur", "coeur", "cardiaque"],
    EMBARAZO: ["enceinte", "grossesse"],
  };
  for (const [code, words] of Object.entries(expected)) {
    for (const word of words) assert.deepEqual(find(`Vous dites ${word} ?`), [code], word);
  }
});

test("the French name of every sign finds it", () => {
  for (const sign of signs) assert.deepEqual(find(sign.fr), [sign.code], sign.fr);
});

test("nothing to show", () => {
  assert.deepEqual(find(""), []);
  assert.deepEqual(find("   "), []);
  assert.deepEqual(find(undefined), []);
  assert.deepEqual(find("Asseyez-vous, s'il vous plaît."), []);
  assert.deepEqual(find("Bonjour, asseyez-vous."), ["TN_3ASLEMA"]);
});

test("a full doctor sentence", () => {
  assert.deepEqual(
    find("Si la fièvre et la toux continuent, prenez ce traitement et revenez en urgence."),
    ["FIEBRE2", "TOSER", "FARMACIA", "URGENTE"],
  );
});

// --- Tunisian everyday signs -----------------------------------------------------------

import { readFileSync } from "node:fs";

const TUNISIAN = JSON.parse(readFileSync(new URL("../../training/tunisian_vocab.json", import.meta.url), "utf8"))
  .map(({ code, fr, ar }) => ({ code, group: "everyday", fr, ar }));
const ALL = [...JSON.parse(readFileSync(new URL("../data/signs.json", import.meta.url), "utf8")).signs
  .filter((s) => !s.code.startsWith("TN_")), ...TUNISIAN];

test("everyday signs keep their place in the sentence", () => {
  const out = signsToSentence(["TN_3ASLEMA", "ENFERMO", "FIEBRE2", "TN_SBITAR"], ALL);
  assert.equal(out.fr, "Bonjour. Je suis malade. J'ai de la fièvre. L'hôpital.");
  assert.ok(out.ar.length > 0);
});

test("every Tunisian sign has a written sentence in both languages", () => {
  const spoken = { TN_OM: "Ma mère.", TN_SBITAR: "L'hôpital.", TN_N3AWNEK: "J'ai besoin d'aide.", TN_T7EB: "Je veux." };
  for (const sign of TUNISIAN) {
    const out = signsToSentence([sign.code], ALL);
    assert.ok(out.fr.endsWith(".") && out.ar.length > 1, `${sign.code}: ${out.fr} / ${out.ar}`);
    if (spoken[sign.code]) assert.equal(out.fr, spoken[sign.code]);
  }
});

test("a yes, a no and a day make short answers", () => {
  assert.equal(signsToSentence(["TN_OUI"], ALL).fr, "Oui.");
  assert.equal(signsToSentence(["TN_NON", "FIEBRE2"], ALL).fr, "Non. J'ai de la fièvre.");
  assert.equal(signsToSentence(["TN_5MIS"], ALL).fr, "Jeudi.");
});

test("a doctor's everyday sentences find the Tunisian signs", () => {
  assert.deepEqual(textToSigns("Bonjour, comment allez-vous ?", ALL), ["TN_3ASLEMA", "TN_LABES"]);
  assert.deepEqual(textToSigns("Allez à l'hôpital aujourd'hui.", ALL), ["TN_SBITAR", "TN_LYOUM"]);
  assert.deepEqual(textToSigns("Revenez jeudi avec votre mère.", ALL), ["TN_5MIS", "TN_OM"]);
  assert.deepEqual(textToSigns("Voulez-vous de l'aide ?", ALL), ["TN_T7EB", "TN_N3AWNEK"]);
  assert.deepEqual(textToSigns("Allez au centre de santé.", ALL), ["TN_MOSTAWSAF"]);
  assert.deepEqual(textToSigns("Votre sœur a de la fièvre ?", ALL), ["TN_O5T", "FIEBRE2"]);
  assert.deepEqual(textToSigns("Oui, d'accord.", ALL), ["TN_OUI"]);
  assert.deepEqual(textToSigns("Voulez-vous que je vous l'écrive ?", ALL), ["TN_T7EB", "TN_NEKTEBLK"]);
});

test("signWords lists the doctor's words that show a sign, its name first", () => {
  const { signWords } = LEXICON;
  assert.deepEqual(signWords("TN_SBITAR", ALL), ["Hôpital", "hôpitaux"]);   // "hôpital", "l'hôpital": same word
  assert.ok(signWords("TOSER", ALL).includes("tousser"));
  assert.deepEqual(signWords("NOT_A_SIGN", ALL), []);
  for (const sign of ALL) {
    for (const word of signWords(sign.code, ALL)) {
      assert.ok(textToSigns(word, ALL).includes(sign.code), `${word} -> ${sign.code}`);
    }
  }
});
