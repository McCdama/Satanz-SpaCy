import requests
import stanza
import spacy_stanza


r = requests.get("http://google.com")       
print(r.status_code)

stanza.download("en")
nlp = spacy_stanza.load_pipeline("en")

doc = nlp("Barack Obama was born in Hawaii. He was elected president in 2008.")
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.dep_, token.ent_type_)
print(doc.ents)




""" Building a Pipeline with a config dictionary --> Maximum customization """
config = {
    'processors': 'tokenize,mwt,pos',
    'lang': 'en',
    'tokenize_model_path': './fr_gsd_models/fr_gsd_tokenizer.pt',
	'mwt_model_path': './fr_gsd_models/fr_gsd_mwt_expander.pt',
	'pos_model_path': './fr_gsd_models/fr_gsd_tagger.pt',
	'pos_pretrain_path': './fr_gsd_models/fr_gsd.pretrain.pt',

    # Use pretokenized text as input & disable tokenization
	'tokenize_pretokenized': True
}

nlp = stanza.Pipeline(**config)
doc = nlp("When Sebastian Thrun started working on self-driving cars at "
        "Google in 2007, few people outside of the company took him "
        "seriously. “I can tell you very senior CEOs of major American "
        "car companies would shake my hand and turn away because I wasn’t "
        "worth talking to,” said Thrun, in an interview with Recode earlier "
        "this week.")
print(doc)