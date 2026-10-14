import spacy
import nltk
from nltk.stem.snowball import SnowballStemmer

english_sentences = [
    "Good morning! How are you feeling today?",
    "Maria travels to London every summer.",
    "The University of Oxford was founded in the twelfth century.",
    "Barack Obama visited Berlin on Monday morning.",
    "The children are playing in the park after school.",
    "Apple produces iPhones in China and India.",
    "I read an interesting book about the history of Rome.",
    "The Danube flows into the Black Sea.",
    "The professor explained the lesson to the first-year students.",
    "Tomorrow we will go to the mountains together.",
]

romanian_sentences = [
    "Bună ziua! Cum te simți astăzi?",
    "Maria călătorește la Londra în fiecare vară.",
    "Universitatea din Oxford a fost fondată în secolul al doisprezecelea.",
    "Barack Obama a vizitat Berlinul luni dimineața.",
    "Copiii se joacă în parc după școală.",
    "Apple produce telefoane în China și India.",
    "Am citit o carte interesantă despre istoria Romei.",
    "Dunărea se varsă în Marea Neagră.",
    "Profesorul a explicat lecția studenților din anul întâi.",
    "Mâine vom merge împreună la munte.",
]


grammar = r"""
  NP: {<DET>?<ADJ>*<NOUN>}   # chunk determiner, adjectives and noun
      {<PROPN>+}             # chunk sequences of proper nouns
"""
chunker = nltk.RegexpParser(grammar)

out = open("results.txt", "w", encoding="utf-8")


def to_nltk(token):
    children = [to_nltk(child) for child in token.children]
    if children:
        return nltk.Tree(token.text, children)
    return token.text


def process(sentences, nlp, stemmer):
    for sentence in sentences:
        doc = nlp(sentence)
        print("\n" + sentence, file=out)

        print("\ntoken, stem, lemma, POS, dependency, head", file=out)
        for token in doc:
            print(token.text, stemmer.stem(token.text), token.lemma_, token.pos_, token.dep_, token.head.text, file=out)

        print("\ndependency tree", file=out)
        for sent in doc.sents:
            to_nltk(sent.root).pretty_print(stream=out)

        print("\nchunks", file=out)
        print(chunker.parse([(token.text, token.pos_) for token in doc]), file=out)

        print("\nentities", file=out)
        for ent in doc.ents:
            print(ent.text, ent.label_, file=out)


print("ENGLISH", file=out)
process(english_sentences, spacy.load("en_core_web_trf"), SnowballStemmer("english"))

print("\n\nROMANIAN", file=out)
process(romanian_sentences, spacy.load("ro_core_news_lg"), SnowballStemmer("romanian"))

out.close()
