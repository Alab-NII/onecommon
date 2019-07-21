from flask import Blueprint, jsonify, render_template, request, redirect, url_for, Markup, Response
from flask import current_app as app

from functools import wraps
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

annotation = Blueprint('annotation', __name__)


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

@annotation.route('/annotation')
@requires_auth
def annotate():
    backend = get_backend()
    chat_data = app.config['chat_data']
    if not request.args.get('chat_id'):
        num_chats = len(chat_data)
        num_success = sum([chat['outcome']['reward'] == 1 for chat in chat_data])
        num_fail = sum([chat['outcome']['reward'] == 0 for chat in chat_data])
        return render_template('simple_chat_list.html',
                                num_chats = num_chats,
                                chat_ids = [chat['uuid'] for chat in chat_data],
                                chat_outcomes = [chat['outcome']['reward'] for chat in chat_data],
                                base_url = 'https://your-original-url/sample/annotation?chat_id=',
                                num_success=num_success,
                                num_fail=num_fail)
    else:
        chat_id = request.args.get('chat_id')
        chat = [chat for chat in chat_data if chat['uuid'] == chat_id]
        chat = chat[0]
        chat_text = ""
        select_id = {}
        for chat_event in chat['events']:
            if chat_event['action'] == 'message':
                chat_text += "{}: {}\n".format(chat_event['agent'], chat_event['data'].encode('ascii', 'ignore'))
            elif chat_event['action'] == 'select':
                chat_text += "<{} selected {}>\n".format(chat_event['agent'], chat_event['data'])
                select_id[chat_event['agent']] = chat_event['data']
        select = {}
        for agent in [0,1]:
            if not agent in select_id:
                select[agent] = None
                continue
            for obj in chat['scenario']['kbs'][agent]:
                #pdb.set_trace()
                if obj['id'] == select_id[agent]:
                    select[agent] = obj
                    break

        backend = get_backend()
        db_path = app.config['user_params']['db']['location']
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''SELECT * FROM survey where chat_id=?''', (chat_id, ))
        res = cursor.fetchone()
        if res:
            survey = True
            cooperative = res[3]
            humanlike = res[4]
            comments = res[5]
        else:
            survey = False
            cooperative = None
            humanlike = None
            comments = None
        conn.close()

        return render_template('simple_visualize.html',
                                chat_id=chat_id,
                                chat_text=chat_text,
                                kb_0=chat['scenario']['kbs'][0],
                                kb_1=chat['scenario']['kbs'][1],
                                select_0=select[0],
                                select_1=select[1],
                                agent_0=chat['agents']['0'],
                                agent_1=chat['agents']['1'],
                                survey=survey,
                                cooperative=cooperative,
                                humanlike=humanlike,
                                comments=comments
                                )

