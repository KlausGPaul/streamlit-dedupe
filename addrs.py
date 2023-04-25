from contextlib import redirect_stdout, redirect_stderr
import io
from math import dist
import sys
import subprocess
import traceback
import streamlit as st
import extra_streamlit_components as stx
from st_aggrid import AgGrid
import pandas as pd
import dedupe
import os
import time
import json
from typing import Tuple
from dedupe._typing import (
    Data,
    Literal,
    RecordDict,
    RecordDictPair,
    RecordID,
    TrainingData,
)

st.set_page_config(layout='wide')

#op_mode = stx.stepper_bar(steps=["Import", "Train", "Cleanse"])
#st.info(f"Phase #{op_mode}")
op_mode = stx.tab_bar(data=[
    stx.TabBarItemData(id=1, title="Import", description=""),
    stx.TabBarItemData(id=2, title="Train", description=""),
    stx.TabBarItemData(id=3, title="Cleanse", description=""),
], default=1)

#st.info(f"{op_mode}")

if 'deduper' not in st.session_state:
    st.session_state['deduper'] = dedupe.Dedupe([{"field":"init","type":"String"}])


def get_records_pair(deduper: dedupe.api.ActiveMatching):
    fields = dedupe.core.unique(var.field for var in deduper.data_model.primary_variables)

    buffer_len = 1  # Max number of previous operations
    unlabeled: list[dedupe._typing.RecordDictPair] = []
    labeled: list[dedupe._typing.LabeledPair] = []

    try:
        if not unlabeled:
            unlabeled = deduper.uncertain_pairs()

        record_pair = unlabeled.pop()
    except IndexError:
        return pd.DataFrame(),pd.DataFrame()

    n_match = len(deduper.training_pairs["match"]) + sum(
        label == "match" for _, label in labeled
    )
    n_distinct = len(deduper.training_pairs["distinct"]) + sum(
        label == "distinct" for _, label in labeled
    )

    adf = []
    for record in record_pair:
        alldata = []
        for field in fields:
            #line = "%s : %s" % (field, record[field])
            alldata.append({"field":field,"value":record[field]})
            #_print(line)
        adf.append(pd.DataFrame(alldata))

    return adf[0],adf[1],labeled,unlabeled,record_pair,n_match,n_distinct

LabeledPair = Tuple[RecordDictPair, dedupe._typing.Literal["match", "distinct", "unsure"]]


def _mark_pair(deduper: dedupe.api.ActiveMatching, labeled_pair: LabeledPair) -> None:
    record_pair, label = labeled_pair
    examples: dedupe._typing.TrainingData = {"distinct": [], "match": []}
    if label == "unsure":
        # See https://github.com/dedupeio/dedupe/issues/984 for reasoning
        examples["match"].append(record_pair)
        examples["distinct"].append(record_pair)
    else:
        # label is either "match" or "distinct"
        examples[label].append(record_pair)
    deduper.mark_pairs(examples)


def process_selection(deduper,selection,labeled,unlabeled,record_pair):
    if selection == "Yes":
        labeled.insert(0, (record_pair, "match"))
        valid_response = True
    elif selection == "No":
        labeled.insert(0, (record_pair, "distinct"))
        valid_response = True
    elif selection == "Unsure":
        labeled.insert(0, (record_pair, "unsure"))
        valid_response = True
    elif selection == "Finished":
        #_print("Finished labeling")
        valid_response = True
        finished = True
    elif selection == "Use Previous":
        valid_response = True
        use_previous = True
        unlabeled.append(record_pair)

    for labeled_pair in labeled:
        dedupe.convenience._mark_pair(deduper, labeled_pair)

    return labeled,unlabeled,record_pair



def streamlit_label(deduper: dedupe.api.ActiveMatching,left_df_display,right_df_display,selection,make_selection) -> None:  # pragma: no cover
    """
    Train a matcher instance (Dedupe, RecordLink, or Gazetteer) from the command line.
    Example
    .. code:: python
       > deduper = dedupe.Dedupe(variables)
       > deduper.prepare_training(data)
       > dedupe.console_label(deduper)
    """
    #global left_df_display,right_df_display,selection#,make_selection

    finished = False
    use_previous = False
    fields = dedupe.core.unique(var.field for var in deduper.data_model.primary_variables)

    buffer_len = 1  # Max number of previous operations
    unlabeled: list[RecordDictPair] = []
    labeled: list[LabeledPair] = []

    counter = 1
    waiting_for_submit = False
    while not finished:
        if not waiting_for_submit:
            print(f"CYCLE {counter}")
            counter += 1
            if use_previous:
                record_pair, _ = labeled.pop(0)
                use_previous = False
            else:
                try:
                    if not unlabeled:
                        unlabeled = deduper.uncertain_pairs()

                    record_pair = unlabeled.pop()
                except IndexError:
                    break

            n_match = len(deduper.training_pairs["match"]) + sum(
                label == "match" for _, label in labeled
            )
            n_distinct = len(deduper.training_pairs["distinct"]) + sum(
                label == "distinct" for _, label in labeled
            )

            adf = []
            for record in record_pair:
                alldata = []
                for field in fields:
                    #line = "%s : %s" % (field, record[field])
                    alldata.append({"field":field,"value":record[field]})
                    #_print(line)
                adf.append(pd.DataFrame(alldata))
                #_print()
            print(f"{n_match}/10 positive, {n_distinct}/10 negative")
            left_df_display.dataframe(adf[0])
            print(end="OK1",file=sys.stderr)
            right_df_display.dataframe(adf[1])
            print(end="OK2",file=sys.stderr)
            waiting_for_submit = True
            print(end="OK3",file=sys.stderr)
        elif make_selection:
            print(f"selection {make_selection} {selection}",file=sys.stderr)
            if selection == "Yes":
                labeled.insert(0, (record_pair, "match"))
                valid_response = True
            elif selection == "No":
                labeled.insert(0, (record_pair, "distinct"))
                valid_response = True
            elif selection == "Unsure":
                labeled.insert(0, (record_pair, "unsure"))
                valid_response = True
            elif selection == "Finished":
                #_print("Finished labeling")
                valid_response = True
                finished = True
            elif selection == "Use Previous":
                valid_response = True
                use_previous = True
                unlabeled.append(record_pair)

            while len(labeled) > buffer_len:
                _mark_pair(deduper, labeled.pop())

            waiting_for_submit = False
        else:
            print(end=".",file=sys.stderr)
            time.sleep(1.0)
        print(end="+",file=sys.stderr)
        time.sleep(1.0)

    for labeled_pair in labeled:
        _mark_pair(deduper, labeled_pair)


df = pd.read_parquet("ops.parquet")


if op_mode == "1":
    #print(f"{op_mode}")
    #st.write("Import")
    columns = df.columns
    types = ["String" for c in columns]
    has_missing = [False for c in columns]
    df_fields = pd.DataFrame({
        "column":columns,
        "type":types,
        "has missing":has_missing
    })
    AgGrid(df_fields)
    

elif op_mode == "2":
    #print(f"{op_mode}")
    data = df.to_dict(orient="index")
    if st.session_state['deduper'].data_model.primary_variables[0].field == "init":
        with st.spinner("Initialising model..."):
            fields = [
            {'field': 'name', 'type': 'String', 'has missing': True},
            {'field': 'address', 'type': 'String', 'has missing': True},
            {'field': 'country_code', 'type': 'Exact', 'has missing': True},
            #{'field': 'Operator', 'type': 'String', 'has missing': True},
            #{'field': 'Address 1', 'type': 'String', 'has missing': True},
            #{'field': 'Address 2', 'type': 'String', 'has missing': True},
            #{'field': 'City', 'type': 'String', 'has missing': True},
            #{'field': 'State', 'type': 'String', 'has missing': True},
            #{'field': 'Zip', 'type': 'Exact', 'has missing': True},
            #{'field': 'Country', 'type': 'String', 'has missing': True}]
            ]
            st.session_state['deduper'] = dedupe.Dedupe(fields)
            print("deduper")

            data = df.to_dict(orient="index")
            if os.path.exists("training.json"):
                with open("training.json","rt") as f:
                    st.session_state['deduper'].prepare_training(data,f)    
                    loaded = True
            else:
                st.session_state['deduper'].prepare_training(data)  
                loaded = False
        #print("prepare_training")
        #st.session_state['deduper'] = True

    #dedupe.console_label = streamlit_label

    selection_container = st.container()
    #st.sidebar.text(loaded)
    train = st.button("Train")
    with st.form(key='my_form',clear_on_submit=True):
        #global make_selection
        left_df_display,selection_container,right_df_display = st.columns(3)
        st.text("Positive")
        match_bar = st.empty()
        st.text("Negative")
        distinct_bar = st.empty()
        with selection_container:
            #global make_selection
            selection = st.radio("Do these records refer to the same thing?",["Yes","No","Unsure","Finished","Use Previous"])
            df_l,df_r,labeled,unlabeled,record_pair,n_match,n_distinct = get_records_pair(st.session_state['deduper'])
            left_df_display.table(df_l)
            right_df_display.table(df_r)
            match_bar.progress(n_match/len(df))
            distinct_bar.progress(n_distinct/len(df))
            make_selection = st.form_submit_button("OK")
            #dedupe.console_label(deduper,left_df_display,right_df_display,selection,make_selection)


    labeled,unlabeled,record_pair = process_selection(st.session_state['deduper'],selection,labeled,unlabeled,record_pair)
    #st.write(labeled)
    #st.write(unlabeled)

    if train:
        with st.spinner("Training Model"):
            st.session_state['deduper'].train()
            with open("training.json", 'w') as tf:
                st.session_state['deduper'].write_training(tf)

            if os.path.exists("training.json"):
                with open("training.json","rt") as f:
                    st.session_state['deduper'].prepare_training(data,f)    
                    loaded = True
            else:
                st.session_state['deduper'].prepare_training(data)  
                loaded = False

        st.sidebar.text(loaded)
        st.write("Clustering...")
        clustered_dupes = st.session_state['deduper'].partition(data, 0.5)

        st.write('# duplicate sets', len(clustered_dupes))

        cluster_membership = {}
        for cluster_id, (records, scores) in enumerate(clustered_dupes):
            for record_id, score in zip(records, scores):
                cluster_membership[int(record_id)] = {
                    "Cluster ID": cluster_id,
                    "confidence_score": score
                }
                cluster_membership[int(record_id)].update(data[record_id])

        #st.code(json.dumps(cluster_membership,indent=2),language="json")
        st.write(clustered_dupes)
        with open("cluster_membership.json","w+t") as f:
            json.dump(cluster_membership,f)

elif op_mode == "3":
    #print(f"{op_mode}")
    pass
