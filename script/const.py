SENT_FEAT = 128
MENTION_TAGS = {"O": 0, "B": 1, "I": 2}
MENTION_ID2TAG = {v: k for k, v in MENTION_TAGS.items()}

# type head
TYPE_NONE = 0
TYPE_EE = 1
TYPE_IE = 2
TYPE_EI = 3
TYPE_II = 4
TYPE_NAMES = {TYPE_NONE: "none", TYPE_EE: "EE", TYPE_IE: "IE", TYPE_EI: "EI", TYPE_II: "II"}
