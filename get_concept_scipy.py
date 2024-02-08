import scispacy
import spacy
from scispacy.abbreviation import AbbreviationDetector
from scispacy.hyponym_detector import HyponymDetector
from scispacy.linking import EntityLinker
import pandas as pd
from collections import defaultdict
from collections import Counter
from tqdm import tqdm
import re

counter = Counter()
def map2_tokenized_pos(text,char_pos):
	pos1=len(text[:char_pos[0]+1].split())-1
	pos2=pos1+len(text[char_pos[0]:char_pos[1]].split())
	return [pos1,pos2]

for fold in [1,2,3,4,5]:
	for dtype in ['test_snippets.csv','train_snippets.csv','val_snippets.csv']:
		# file_path = 'fold'+str(fold)+'/'+dtype+'_with_MT.csv'
		# data = pd.read_csv('concepts_0406/'+file_path)
		file_path = '../clinicalBERT-master/data/3days/fold'+str(fold)+'/'+dtype
		data = pd.read_csv(file_path)


		# data = pd.read_csv('concepts_0406/fold3/test_with_MT.csv')
		# data = data[:30]

		nlp = spacy.load("en_core_sci_md")
		# nlp.add_pipe("abbreviation_detector")
		# nlp.add_pipe("hyponym_detector", last=True, config={"extended": False})
		nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
		linker = nlp.get_pipe("scispacy_linker")

		# text = """
		# date of birth: sex: f service: medicine allergies: haldol attending: chief complaint: delta ms, lethargy, ?sepsis . major surgical or invasive procedure: none history of present illness: hx obtained per ed notes and sister . hpi: 35f with disease who presented today from daycare after her healthcare providers noted that she was lethargic. they were initially unable to obtain a blood pressure. the patient was noted to have a very rapid heart rate. vitals were finally obtained and were as follows: bp 70/50 (baseline sbps 80-90), hr 113, o2 sat 99% on 3l nc. . the patient was transferred to where she was noted to have a temp of 4, hr 200 and sbp 80s. ekg was noteworthy for a wide complex tachycardia. the patient received adenosine 6mg and then 12mg with no improvement. she was cardioverted into sinus rhythm. her d-dimer was elevated at 3590, lactate was 5 and trop t 39 in the setting of renal insufficiency. a ct-a was negative for a pe. the patient was transferred to the micu for further mgmt. . past medical history: disease anemia nonverbal at baseline . social history: meds: tylenol ensure . sochx: patient lives at home with sister and brother. she also goes to daycare. she is non-verbal at baseline. . family history: father who passed away of dz physical exam: t 7, hr 65-68, bp 91-97/61-63, r 14-21, o2 sat 100%2l gen: thin appearing female lying in fetal position in nad heent: mm dry, op clear heart: nl rate, s1s2, no gmr lungs: cta b/l abd: flat, soft, nt, nd, +bs, negative guardin, negative rebound tenderness ext: wwp, +dp b/l neuro: unable to assess . pertinent results: ct-a impression: no evidence of pulmonary embolism. poorly defined opacities within the lungs bilaterally, possibly representing combination of atelectasis and consolidation. air bronchograms in the right middle lobe suggests possible infection. . cxr impression: left lower lobe process suggesting
		# """
		# doc = nlp(text)
		tui_exclude = ['T040', 'T170', 'T080', 'T053', 'T032', 'T057', 'T073', 'T078', 'T058', 'T097', 'T079', 'T090', 'T077', 'T099', 'T092', 'T185', 'T098', 'T016', 'T083', 'T067', 'T015', 'T021', 'T056', 'T100', 'T064', 'T051', 'T087', 'T102', 'T075', 'T095', 'T094']

		record = defaultdict(list)
		data['MT'] = None
		data['dependency'] = None
		# data['sentence_index'] = None
		data['sentence_attribute'] = None
		# data['mt_indices'] = None
		removed_mt = defaultdict(list)

		for index,row in tqdm(data.iterrows()):
			text = row['TEXT']
			text=re.sub(r'[. ]+','. ',text)
			doc = nlp(text)
			extracted_mt = []
			extracted_dep = []
			sentence_idx = []
			sentence_attribute = []
			mt_indices = []
			flag = 0
			for sent_i, sent in enumerate(doc.sents):
				sentence_attr = [sent_i, sent.start, sent.text, [], None]
				entities = sent.ents
				mt_index_list = []
				for ent in entities:
					if len(ent._.kb_ents)>0:
						# print('ent._.kb_ents',ent._.kb_ents)
						umls_ent = ent._.kb_ents[0]

						# for umls_ent in ent._.kb_ents:
						e, start, end, tok_start, tok_end = ent.text, ent.start_char, ent.end_char, ent.start, ent.end
						cui_code, normed_name, aliases, tui_code, definition = linker.kb.cui_to_entity[umls_ent[0]]
						# tok_start_ma, tok_end_ma = map2_tokenized_pos(text,[start, end])
						# print('check', tok_start , tok_end, tok_start_ma, tok_end_ma)
						# print(doc[tok_start:tok_end], text.split()[tok_start_ma:tok_end_ma])
						# for x in tui_code:
						# 	if x in tui_exclude:
						# 		removed_mt[x].append(e)
						tui_code = [x for x in tui_code if x not in tui_exclude]
						
						if len(tui_code)>0:
							flag = 1
							extracted_mt.append((e, normed_name, [tok_start, tok_end], [start, end], cui_code, tui_code))
							sentence_attr[3].append((e, normed_name, [tok_start, tok_end], [start, end], cui_code, tui_code))
							mt_index_list.extend(list(range(tok_start, tok_end)))
							# for t_code in tui_code:
							# 	record[t_code].append(e)
							# print('sent', sent)
							dependency = defaultdict(list)
							for token in sent:
								t_start = token.i-sent.start

								extracted_dep.append((token.dep_, token.text, token.head.text, t_start, token.head.i-sent.start, token.pos_))
								dependency[t_start].append((token.dep_, token.text, token.head.text, t_start, token.head.i-sent.start, token.pos_, token.head.pos_))
							sentence_attr[4] = dict(dependency)
				sentence_attr.append(mt_index_list)
				sentence_attribute.append(sentence_attr)
			data.at[index, 'MT'] = extracted_mt
			data.at[index, 'dependency'] = extracted_dep
			data.at[index, 'sentence_attribute'] = sentence_attribute
		data = data[['ID','TEXT','Label','MT','dependency','sentence_attribute']]

		data.to_csv('pyg_data_3days/'+ 'fold'+str(fold)+'/'+dtype)