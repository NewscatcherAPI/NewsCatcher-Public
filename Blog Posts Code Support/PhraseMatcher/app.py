import streamlit as st
from spacy import displacy
import spacy
from spacy.matcher import PhraseMatcher
from cleanco import basename

import streamlit as st
import streamlit.components.v1 as components



import pandas as pd


st.set_page_config(
   page_title="Using PhraseMatcher To Label Data",
   page_icon="🛠",
   layout="wide",
   initial_sidebar_state="expanded",
   menu_items={
        "Get help": None,
        "Report a Bug": None,
        "About": None
            }
)

st.title("Using PhraseMatcher To Label Data")


data = pd.read_csv("nasdaq-listed-symbols.csv")

data['Cleaned Name'] = data['Company Name'].apply(basename)
data['Cleaned Name'] = data['Cleaned Name'].apply(basename)

names = pd.concat([data['Company Name'], data['Cleaned Name']], ignore_index = True).drop_duplicates()

names = [name for name in names if name != " " and len(name) > 0]

name_corrections = {"A": "A-Mark", "Federal": "Federal-Mogul",
                    "Global": "Global-Tech Advanced Innovations",
                    "G": "G-III Apparel", "Heritage": "Heritage Crystal Clean",
                    "II": "II-VI", "Mid": "Microchip Technology", 
                    "Pro":"Pro-Dex", "Perma":"Perma-Fix Environmental Services",
                    "Park": "Park-Ohio Holdings", "Bio": "Bio-Techne",
                    "ROBO": " ROBO Global Robotics and Automation Index ETF",
                    "United": "United-Guardian", "Uni":"Uni-Pixel"}

names = [name_corrections[name] if name in name_corrections.keys() else name for name in names ]                    

nlp = spacy.load("en_core_web_sm")

matcher = PhraseMatcher(nlp.vocab)


patterns = [nlp.make_doc(name) for name in names]
matcher.add("ORG", patterns)

patterns = [nlp.make_doc(symbol) for symbol in data['Symbol']]
matcher.add("TICKER", patterns)

text = st.text_area("Enter the text you want to label", help = 'Text to label',
                        placeholder = 'Enter the text you want to label', height = 200)


st.sidebar.markdown("### What Do You Want To Match For?")
match_org = st.sidebar.checkbox('Organization Names', value = True, help = "Do you want to match for organization names?")
match_ticker = st.sidebar.checkbox('Ticker Symbols', value = True, help = "Do you want to \
        match for organization ticker symbols?")


if len(text) == 0:
    st.markdown("### Please Enter Some Text To Match For 🤗")
else:
    doc = nlp(text)
    matches = matcher(doc)
 
    if len(matches) == 0:
        st.markdown("### No Organization(s) Found 😥")
    else:
        # displacy options
        colors = {"ORG": "#F67DE3", "TICKER": "#7DF6D9"}
        options = {"colors": colors}

        plot_data = {
                "text": doc.text,
                "ents": [],
                "title": None
            }
        with st.spinner("Labelling Text..."):
            matches_with_dup = {"ORG":{}, "TICKER": {}}
            for match_id, span_start, span_end in matches:

                rule_id = nlp.vocab.strings[match_id]
                text = doc[span_start: span_end].text
                start_idx = doc.text.index(doc[span_start].text)
                end_idx = start_idx + len(text)

                if match_org and match_ticker:
                    matches_with_dup[rule_id][text] = {"start": start_idx, "end": end_idx, "label": rule_id}
                elif match_org and not match_ticker:
                    if rule_id == "ORG":
                        matches_with_dup[rule_id][text] = {"start": start_idx, "end": end_idx, "label": rule_id}
                elif not match_org and match_ticker:
                    if rule_id == "TICKER":
                        matches_with_dup[rule_id][text] = {"start": start_idx, "end": end_idx, "label": rule_id}

            # substring names/symbols will appear multiple times but the expanded 
            # longest versions of the names/symbols will appear only once    

            for ent_type in matches_with_dup.keys():
                matches = matches_with_dup[ent_type]
                keys = matches.keys()
                counts = {text:0 for text in keys}
                for text in keys:
                    for key in keys:
                        if text in key:
                            counts[text] += 1
                for text, count in counts.items():
                    if count == 1:
                        plot_data['ents'].append(matches[text])

            plot_data['ents'] = sorted(plot_data['ents'], key = lambda ent: ent['start'])

            st.markdown("### Labeled Data")
            html = displacy.render(plot_data , style="ent", options=options, manual=True, page =True)
            components.html(html, height = 500, scrolling  = True)