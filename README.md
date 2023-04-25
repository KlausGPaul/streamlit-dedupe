# streamlit-dedupe
A rather rough PoC for wrapping [dedupe](https://github.com/dedupeio/dedupe) active learning console inside a streamline app. The purpose is
to make the training available to a group of users who lack access to the infrastructure and
data required to set up and perform dedupe interactive labelling.

You will need a file "ops.parquet" which contains the entries you want to deduplicate. Also, at the moment,
the implementation uses a static schema.

The output is a labelled json file (which can be used to (re)train a dedupe model.
