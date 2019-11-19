from flask import Blueprint, jsonify, render_template, request, redirect, url_for, Markup, Response
from flask import current_app as app

from functools import wraps
import itertools
import json
import sqlite3
import pdb

from collections import defaultdict
import operator

from cocoa.web.views.utils import userid, format_message
from cocoa.web.main.utils import Status
from cocoa.core.event import Event

from main.db_reader import DatabaseReader

from web.main.backend import Backend
get_backend = Backend.get_backend

coreference = Blueprint('coreference', __name__)

def check_auth(username, password):
    """This function is called to check if a username /
    password combination is valid.
    """
    return username == 'sample' and password == 'sample'

def authenticate():
    """Sends a 401 response that enables basic auth"""
    return Response(
    'Could not verify your access level for that URL.\n'
    'You have to login with proper credentials', 401,
    {'WWW-Authenticate': 'Basic realm="Login Required"'})

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

@coreference.route('/_finish_annotation/', methods=['POST'])
def finish_annotation():
    data = request.get_json(force=True)
    chat_id = data['chat_id']
    annotation_data = data['annotation_data']
    worker_id = data['worker_id']
    assignment_id = data['assignment_id']

    if worker_id == "aggregated":
        worker_id = "admin" # consider it as admin

    if not chat_id in app.config['referent_annotation']:
        app.config['referent_annotation'][chat_id] = {}
    app.config['referent_annotation'][chat_id][worker_id] = {} # we are overwriting with the new annotation
    for markable_id in list(annotation_data.keys()):
        app.config['referent_annotation'][chat_id][worker_id][markable_id] = {}
        if 'ambiguous' not in annotation_data[markable_id]:
            pdb.set_trace()
        app.config['referent_annotation'][chat_id][worker_id][markable_id]['ambiguous'] = annotation_data[markable_id]['ambiguous']
        app.config['referent_annotation'][chat_id][worker_id][markable_id]['unidentifiable'] = annotation_data[markable_id]['unidentifiable']
        app.config['referent_annotation'][chat_id][worker_id][markable_id]['referents'] = annotation_data[markable_id]['referents']
        app.config['referent_annotation'][chat_id][worker_id][markable_id]['assignment_id'] = assignment_id # add assignment id

    if len(app.config['referent_annotation'][chat_id][worker_id].keys()) > 0:
        return jsonify(success=True)
    else:
        print("finish annotation failed! worker: {}".format(worker_id))
        del app.config['referent_annotation'][chat_id][worker_id]
        return jsonify(success=False)

@coreference.route('/_accept_annotation/', methods=['POST'])
def accept_annotation():
    data = request.get_json(force=True)
    chat_id = data['chat_id']
    annotation_data = data['annotation_data']
    worker_id = data['worker_id']
    assignment_id = data['assignment_id']
    accept_message = data['accept_message']

    for markable_id in list(annotation_data.keys()):
        app.config['referent_annotation'][chat_id][worker_id][markable_id]["accept_message"] = accept_message

    return jsonify(success=True)

@coreference.route('/_reject_annotation/', methods=['POST'])
def reject_annotation():
    data = request.get_json(force=True)
    chat_id = data['chat_id']
    annotation_data = data['annotation_data']
    worker_id = data['worker_id']
    assignment_id = data['assignment_id']
    reject_message = data['reject_message']

    if not chat_id in app.config['rejected_referent_annotation']:
        app.config['rejected_referent_annotation'][chat_id] = {}
    # pop annotation and move to rejected
    annotation_data = app.config['referent_annotation'][chat_id].pop(worker_id, None)
    if annotation_data:
        app.config['rejected_referent_annotation'][chat_id][worker_id] = annotation_data

    for markable_id in list(annotation_data.keys()):
        app.config['rejected_referent_annotation'][chat_id][worker_id][markable_id]["reject_message"] = reject_message


    return jsonify(success=True)


def is_annotatable_markable(markable):
    if markable["generic"] or markable["no-referent"] or markable["all-referents"] or markable["anaphora"] or markable["cataphora"] or markable["predicative"]:
        return False
    else:
        return True

def calculate_statistics(chat_ids, markable_annotation, referent_annotation, chat_data, mturk_only=False):
    num_markables = []
    num_ambiguous = []
    num_unidentifiable = []
    num_no_referent = []
    observed_agreement = []
    exact_match_rate = []
    for chat_id in chat_ids:
        markables = markable_annotation[chat_id]["markables"]

        if not chat_id in referent_annotation:
            annotators = []
        elif mturk_only:
            annotators = [x for x in list(referent_annotation[chat_id].keys()) if x.startswith("MT_")]
        else:
            annotators = list(referent_annotation[chat_id].keys())

        chat = [chat for chat in chat_data if chat['uuid'] == chat_id]
        chat = chat[0]
        agent_0_kb, agent_1_kb = chat["scenario"]["kbs"]
        agent_0_ents = ["agent_0_{}".format(x['id']) for x in agent_0_kb]
        agent_1_ents = ["agent_1_{}".format(x['id']) for x in agent_1_kb]

        # statistics
        annotatable_markables = 0
        ambiguous = 0
        unidentifiable = 0
        no_referent = 0
        pairwise_judgements = 0
        pairwise_agreement = 0
        exact_match = 0
        exact_match_judgements = 0
        error = False
        for markable in markables:
            if is_annotatable_markable(markable):
                annotatable_markables += 1
            else:
                continue
            markable_id = markable["markable_id"]
            speaker = markable["speaker"]
            candidate_ents = agent_0_ents if speaker == 0 else agent_1_ents
            for annotator in annotators:
                if not markable_id in referent_annotation[chat_id][annotator]:
                    print("markable not found!! chat_id {} | annotator {} | markable_id {}".format(chat_id, annotator, markable_id))
                    error = True
                    pdb.set_trace()
                    break
                if referent_annotation[chat_id][annotator][markable_id]["ambiguous"]:
                    ambiguous += 1
                if referent_annotation[chat_id][annotator][markable_id]["unidentifiable"]:
                    unidentifiable += 1
                if len(referent_annotation[chat_id][annotator][markable_id]["referents"]) == 0:
                    no_referent += 1
            if error:
                break
            for ent in candidate_ents:            
                is_referent = 0
                is_not_referent = 0
                for annotator in annotators:
                    if ent in referent_annotation[chat_id][annotator][markable_id]["referents"]:
                        is_referent += 1
                    else:
                        is_not_referent += 1
                pairwise_agreement += is_referent * (is_referent - 1) / 2
                pairwise_agreement += is_not_referent * (is_not_referent - 1) / 2
                pairwise_judgements += len(annotators) * (len(annotators) - 1) / 2

            for a, b in itertools.combinations(annotators, 2):
                exact_match_judgements += 1
                if set(referent_annotation[chat_id][a][markable_id]["referents"]) == set(referent_annotation[chat_id][b][markable_id]["referents"]):
                    exact_match += 1

        if error:
            num_markables.append(0)        
            num_ambiguous.append(0)
            num_unidentifiable.append(0)
            num_no_referent.append(0)
            observed_agreement.append(None)
            exact_match_rate.append(None)
        else:
            num_markables.append(annotatable_markables)        
            num_ambiguous.append(ambiguous)
            num_unidentifiable.append(unidentifiable)
            num_no_referent.append(no_referent)
            if pairwise_judgements == 0:
                observed_agreement.append(None)
                exact_match_rate.append(None)
            else:
                observed_agreement.append(round(1.0 * pairwise_agreement / pairwise_judgements, 3))
                exact_match_rate.append(round(1.0 * exact_match / exact_match_judgements, 3))

    return num_markables, num_ambiguous, num_unidentifiable, num_no_referent, observed_agreement, exact_match_rate

@coreference.route('/coreference')
def coreference_annotation():
    chat_data = app.config['chat_data']
    markable_annotation = app.config['markable_annotation']
    batch_info = app.config['batch_info']
    referent_annotation = app.config['referent_annotation']
    aggregated_referent_annotation = app.config['aggregated_referent_annotation']
    model_referent_annotation = app.config['model_referent_annotation']

    #request args
    hit_id = request.args.get('hitId', default=None, type=None)
    assignment_id = request.args.get('assignmentId', default="admin", type=None)
    turk_submit_to = request.args.get('turkSubmitTo', default=None, type=None)
    worker_id = request.args.get('workerId', default="admin", type=None)
    batch_id = request.args.get('batch_id', default=None, type=None)

    mturk = True if hit_id else None
    if mturk:
        worker_id = "MT_" + worker_id

    if not mturk and not request.args.get('chat_id'):
        if batch_id:
            chat_ids = batch_info[batch_id]
            batch_url = "&batch_id={}".format(batch_id)
        else:
            chat_ids = list(markable_annotation.keys())
            batch_url = ""
        num_chats = len(chat_ids)
        admin_url = []
        worker_urls = []
        aggregated_url = []
        model_url = []
        for chat_id in chat_ids:
            _worker_urls = []
            if chat_id in referent_annotation:
                annotators = list(referent_annotation[chat_id].keys())
                for annotator in annotators:
                    if annotator == "admin":
                        admin_url.append("&workerId=admin")
                    elif annotator.startswith("MT_"):
                        if len(referent_annotation[chat_id][annotator].keys()) == 0:
                            print("empty annotation found!")
                            _worker_urls.append("")
                        else:
                            first_markable_id = list(referent_annotation[chat_id][annotator].keys())[0]
                            _assignment_id = referent_annotation[chat_id][annotator][first_markable_id]["assignment_id"]
                            _worker_urls.append("&workerId={}&assignmentId={}".format(annotator, _assignment_id))
                    else:
                        _worker_urls.append("")
                if not "admin" in annotators:
                    admin_url.append("")
            else:
                admin_url.append("")
                _worker_urls = ["", "", ""]
            worker_urls.append(_worker_urls)
            if chat_id in aggregated_referent_annotation:
                aggregated_url.append("&workerId=aggregated")
            else:
                aggregated_url.append("")
            if chat_id in model_referent_annotation:
                model_url.append("&workerId=model")
            else:
                model_url.append("")
        
        num_markables, num_ambiguous, num_unidentifiable, num_no_referent, observed_agreement, exact_match_rate = calculate_statistics(chat_ids, markable_annotation, referent_annotation, chat_data)

        return render_template('coreference_list.html',
                                num_chats = num_chats,
                                chat_ids = chat_ids,
                                base_url = 'http://localhost:5000/sample/coreference',
                                admin_url = admin_url,
                                worker_urls = worker_urls,
                                aggregated_url = aggregated_url,
                                model_url = model_url,
                                batch_url = batch_url,
                                num_markables = num_markables,
                                num_ambiguous = num_ambiguous,
                                num_unidentifiable = num_unidentifiable,
                                num_no_referent = num_no_referent,
                                observed_agreement = observed_agreement,
                                exact_match_rate = exact_match_rate)

    else:
        if mturk:
            if not request.args.get('chat_id'):
                return render_template('not_annotatable.html')
            chat_id = request.args.get('chat_id')
            if chat_id in referent_annotation and worker_id in referent_annotation[chat_id]:
                return render_template('not_annotatable.html')
        else:
            chat_id = request.args.get('chat_id')

        chat = [chat for chat in chat_data if chat['uuid'] == chat_id]
        chat = chat[0]
        text = markable_annotation[chat_id]["text"]
        markup_text = ""

        starts = []
        ends = []
        speakers = []
        markable_ids = []
        for i, markable in enumerate(markable_annotation[chat_id]["markables"]):
            if is_annotatable_markable(markable):
                starts.append(int(markable["start"]))
                ends.append(int(markable["end"]))
                speakers.append(int(markable["speaker"]))
                markable_ids.append(markable["markable_id"])

        for i, ch in enumerate(text):
            if i in starts:
                markable_idx = starts.index(i)
                markup_text += "<span class=\"markable speaker_{1}\" id=\"{0}\" onclick=\"selectMarkable(\'{0}\')\">".format(markable_ids[markable_idx], speakers[markable_idx])
            elif i in ends:
                markup_text += "</span>"
            markup_text += ch
        if len(text) in ends:
            markup_text += "</span>"

        select_id = {}
        for chat_event in chat['events']:
            if chat_event['action'] == 'select':
                select_id[chat_event['agent']] = chat_event['data']

        select = {}
        for agent in [0,1]:
            if not agent in select_id:
                select[agent] = None
                continue
            for obj in chat['scenario']['kbs'][agent]:
                if obj['id'] == select_id[agent]:
                    select[agent] = obj
                    break

        utterances = []

        for utterance in markup_text.split("\n"):
            if utterance.startswith("0:"):
                markup_utterance = Markup("<p class=\"kaiwa-text\">" + utterance + "</p>")
                utterances.append(markup_utterance)
                #markup_utterance = Markup("<div class=\"kaiwa-text-left\"><p class=\"kaiwa-text\">" + utterance + "</p></div>")
                #utterances.append(markup_utterance)
            else:
                markup_utterance = Markup("<p class=\"kaiwa-text\">" + utterance + "</p>")
                utterances.append(markup_utterance)

        #markup_text = markup_text.replace("\n", "<br/>")
        #markup_text = Markup(markup_text)

        if chat_id in app.config['referent_annotation'] and worker_id in app.config['referent_annotation'][chat_id]:
            annotation_data = app.config['referent_annotation'][chat_id][worker_id]
        elif chat_id in app.config['aggregated_referent_annotation'] and worker_id == "aggregated":
            annotation_data = {}
            for markable_id in markable_ids:
                annotation_data[markable_id] = app.config['aggregated_referent_annotation'][chat_id][markable_id] 
        elif chat_id in app.config['model_referent_annotation'] and worker_id == "model":
            annotation_data = {}
            for markable_id in markable_ids:
                annotation_data[markable_id] = app.config['model_referent_annotation'][chat_id][markable_id]
        else:
            annotation_data = {}

        if batch_id:
            return_url = 'http://localhost:5000/sample/coreference?batch_id={}'.format(batch_id)
        else:
            return_url = 'http://localhost:5000/sample/coreference'

        return render_template('coreference.html',
                                chat_id=chat_id,
                                utterances=utterances,
                                kb_0=chat['scenario']['kbs'][0],
                                kb_1=chat['scenario']['kbs'][1],
                                select_0=select[0],
                                select_1=select[1],
                                instructions=Markup(app.config['coreference_instructions']),
                                markable_ids=markable_ids,
                                turk_submit_to=request.args.get('turkSubmitTo', default=None, type=None),
                                worker_id=worker_id,
                                assignment_id=assignment_id,
                                annotation_data=annotation_data,
                                mturk=mturk,
                                return_url=return_url
                                )
