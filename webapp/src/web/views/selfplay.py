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

selfplay = Blueprint('selfplay', __name__)

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


def is_annotatable_markable(markable):
    if markable["generic"] or markable["no-referent"] or markable["all-referents"] or markable["anaphora"] or markable["cataphora"] or markable["predicative"]:
        return False
    else:
        return True

def calculate_statistics(scenario_ids, selfplay_markables, selfplay_referents):
    num_markables = []
    num_referents = []
    results = []

    for scenario_id in scenario_ids:
        markables = selfplay_markables[scenario_id]['markables']
        num_markables.append(len(markables))

        num_referents_dict = defaultdict(lambda: 0)

        for markable in markables:
            markable_id = markable['markable_id']
            referents = selfplay_referents[scenario_id][markable_id]
            num_referents_dict[len(referents)] += 1

        num_referents.append(num_referents_dict)

        results.append(int(selfplay_markables[scenario_id]['selections'][0] == selfplay_markables[scenario_id]['selections'][1]))

    return num_markables, num_referents, results

@selfplay.route('/selfplay')
def selfplay_coreference_annotation():
    selfplay_scenarios = app.config['selfplay_scenarios']
    selfplay_markables = app.config['selfplay_markables']
    selfplay_referents = app.config['selfplay_referents']

    #request args
    scenario_id = request.args.get('scenario_id', default=None, type=None)

    if not scenario_id:
        scenario_ids = list(selfplay_markables.keys())

        num_markables, num_referents, results = calculate_statistics(scenario_ids, selfplay_markables, selfplay_referents)

        return render_template('selfplay_scenario_list.html',
                                num_scenarios = len(scenario_ids),
                                scenario_ids = scenario_ids,
                                base_url = 'http://localhost:5000/sample/selfplay',
                                num_markables = num_markables,
                                num_referents = num_referents,
                                results = results)
    else:
        text = selfplay_markables[scenario_id]["text"]
        markup_text = ""

        scenario = [scenario for scenario in selfplay_scenarios if scenario['uuid'] == scenario_id]
        scenario = scenario[0]

        starts = []
        ends = []
        speakers = []
        markable_ids = []
        for i, markable in enumerate(selfplay_markables[scenario_id]["markables"]):
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

        """
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
        """
        utterances = []

        for utterance in markup_text.split("\n"):
            if utterance.startswith("0:"):
                markup_utterance = Markup("<p class=\"kaiwa-text\">" + utterance + "</p>")
                utterances.append(markup_utterance)
            else:
                markup_utterance = Markup("<p class=\"kaiwa-text\">" + utterance + "</p>")
                utterances.append(markup_utterance)

        if len(selfplay_referents[scenario_id]) > 0:
            referents = selfplay_referents[scenario_id]
        else:
            referents = {}

        select = {}
        for agent in [0,1]:
            for obj in scenario['kbs'][agent]:
                if obj['id'] == selfplay_markables[scenario_id]['selections'][agent]:
                    select[agent] = obj
                    break

        return render_template('selfplay.html',
                                scenario_id=scenario_id,
                                utterances=utterances,
                                kb_0=scenario['kbs'][0],
                                kb_1=scenario['kbs'][1],
                                select_0=select[0],
                                select_1=select[1],
                                instructions=Markup(app.config['coreference_instructions']),
                                markable_ids=markable_ids,
                                markables=selfplay_markables[scenario_id],
                                referents=referents
                                )
