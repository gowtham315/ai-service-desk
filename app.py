import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import database as db
from train import clean_text

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"

st.set_page_config(page_title="AI IT Service Desk", layout="wide")

# Initialize Database
db.init_db()

@st.cache_resource
def load_artifacts():
    vectorizer_path = MODEL_DIR / "tfidf_vectorizer.pkl"
    category_model_path = MODEL_DIR / "category_model.pkl"
    priority_model_path = MODEL_DIR / "priority_model.pkl"
    if not all(path.is_file() for path in (vectorizer_path, category_model_path, priority_model_path)):
        return None, None, None
        
    with vectorizer_path.open('rb') as f:
        vec = pickle.load(f)
    with category_model_path.open('rb') as f:
        cat_m = pickle.load(f)
    with priority_model_path.open('rb') as f:
        prio_m = pickle.load(f)
    return vec, cat_m, prio_m

vectorizer, cat_model, prio_model = load_artifacts()

st.title("🤖 AI-Based IT Service Desk Triage System")
st.markdown("Automated ticket classification and priority prediction system for IT support teams.")

if vectorizer is None:
    st.error("Model files not found! Please run `python train.py` first to generate models.")
    st.stop()

tab1, tab2 = st.tabs(["Submit & Classify Ticket", "View Ticket History"])

with tab1:
    st.subheader("Ticket Input")
    ticket_text = st.text_area("Enter IT Support Ticket Description:", height=150, 
                               placeholder="e.g., Cannot access the main production database server due to network timeout.")
    
    if st.button("Analyze Ticket", type="primary"):
        if not ticket_text.strip():
            st.warning("Please provide a valid ticket description.")
        else:
            cleaned = clean_text(ticket_text)
            if not cleaned:
                st.warning("Please enter a description containing letters or numbers.")
            else:
                features = vectorizer.transform([cleaned])
            
                # Predict Category
                cat_pred = cat_model.predict(features)[0]
                cat_probs = cat_model.predict_proba(features)[0]
                cat_conf = float(np.max(cat_probs)) * 100
            
                # Predict Priority
                prio_pred = prio_model.predict(features)[0]
                prio_probs = prio_model.predict_proba(features)[0]
                prio_conf = float(np.max(prio_probs)) * 100
            
                # Log to Database
                ticket_id = db.log_ticket(ticket_text, cat_pred, cat_conf, prio_pred, prio_conf)
            
                st.success(f"Ticket #{ticket_id} classified successfully.")
            
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Predicted Category", value=cat_pred)
                    st.write(f"**Confidence:** {cat_conf:.2f}%")
            
                with col2:
                    st.metric(label="Predicted Priority", value=prio_pred)
                    st.write(f"**Confidence:** {prio_conf:.2f}%")

with tab2:
    st.subheader("Logged Ticket Records")
    records = db.get_all_tickets()
    if records:
        records_by_id = {record[0]: record for record in records}
        ticket_ids = list(records_by_id)
        selected_ticket_id = st.selectbox(
            "Ticket to review",
            ticket_ids,
            format_func=lambda ticket_id: (
                f"#{ticket_id} - {records_by_id[ticket_id][2][:80]}"
            ),
        )
        selected_ticket = records_by_id[selected_ticket_id]
        category_options = cat_model.classes_.tolist()
        priority_options = prio_model.classes_.tolist()
        category_default = (
            selected_ticket[7]
            if selected_ticket[7] in category_options
            else selected_ticket[3]
        )
        priority_default = (
            selected_ticket[8]
            if selected_ticket[8] in priority_options
            else selected_ticket[5]
        )

        with st.form(f"ticket_review_{selected_ticket_id}"):
            corrected_category = st.selectbox(
                "Confirmed category",
                category_options,
                index=category_options.index(category_default),
            )
            corrected_priority = st.selectbox(
                "Confirmed priority",
                priority_options,
                index=priority_options.index(priority_default),
            )
            save_review = st.form_submit_button("Save review")

        if save_review:
            if db.save_correction(selected_ticket_id, corrected_category, corrected_priority):
                st.success("Agent feedback saved for future training.")
            else:
                st.error("That ticket could not be updated. Refresh the history and try again.")

        records = db.get_all_tickets()
        df_records = pd.DataFrame(
            records,
            columns=[
                "ID",
                "Timestamp",
                "Description",
                "Predicted Category",
                "Category Conf (%)",
                "Predicted Priority",
                "Priority Conf (%)",
                "Confirmed Category",
                "Confirmed Priority",
                "Reviewed At",
            ],
        )
        st.dataframe(df_records, width="stretch")
    else:
        st.info("No ticket logs found in the database.")
        
        #python -m streamlit run app.py
        