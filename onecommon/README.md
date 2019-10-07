# Find One In Common!

**Find One In Common!** is a common grounding dialogue task built on StanfordNLP's [CoCoA framework](https://github.com/stanfordnlp/cocoa).
This repository includes basic functions for dataset collection, dataset visualization and model development.

To run the web server, move to `src` directory and simply use `run_sample.sh`. Make sure to append the PYTHONPATH to `onecommon/onecommon` and `onecommon/onecommon/src`, e.g.

```
export PYTHONPATH="~/onecommon/onecommon/src:~/onecommon/onecommon:${PYTHONPATH}"
```

By default, <http://localhost:5000/sample> can be used for dataset collection. When more than two people are connected to this URL, we create pairs to start playing the dialogue task. 

After collecting the dialogues, you can use <http://localhost:5000/sample/admin> to view the collected dialogues and decide whether to accept or reject the dialogues. All the dialogues collected in the AAAI paper can be seem from the URL <http://localhost:5000/sample/annotation>. By default, these URLs are password protected (username is *sample* and password is *sample*).