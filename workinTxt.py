import textacy
import textacy.datasets
# import numpy
# import spacy
#
# text = ('Since the so-called "statistical revolution" in the late 1980s and mid 1990s, '
#         'much Natural Language Processing research has relied heavily on machine learning. '
#         'Formerly, many language-processing tasks typically involved the direct hand coding '
#         'of rules, which is not in general robust to natural language variation. '
#         'The machine-learning paradigm calls instead for using statistical inference '
#         'to automatically learn such rules through the analysis of large corpora '
#         'of typical real-world examples.')
#
# textacy.text_utils.KWIC(text, 'example', window_width=35)
#
# print(textacy.preprocess_text(text, lowercase=True, no_punct=True)+"\n")
# # spacy.load('en')
# doc = textacy.Doc(text)

cw = textacy.datasets.CapitolWords()
cw.download()
records = cw.records(speaker_name={'Hillary Clinton', 'Barack Obama'})
text_stream, metadata_stream = textacy.fileio.split_record_fields(records, 'text')
corpus = textacy.Corpus('en', texts=text_stream, metadatas=metadata_stream)
print(corpus)
