from logo_clf.metric import CODE_L_TO_LABEL_L
from logo_clf.utils import read_json

CODE_S_TO_LABEL_S = read_json("code_s_to_label_s.json")
CODE_M_TO_LABEL_M = read_json("code_m_to_label_m.json")
CODE_L_TO_LABEL_L = read_json("code_l_to_label_l.json")

LABEL_S_TO_CODE_S = read_json("label_s_to_code_s.json")
LABEL_S_TO_DESC_S_KO = read_json("label_s_to_desc_s_ko.json")