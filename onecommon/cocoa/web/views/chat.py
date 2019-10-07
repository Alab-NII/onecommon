import time
import pdb
from flask import Blueprint, jsonify, render_template, request, redirect, url_for, Markup
from flask import current_app as app

from cocoa.web.views.utils import generate_userid, userid, format_message
from cocoa.web.main.utils import Status
from cocoa.core.event import Event

from web.main.backend import Backend
get_backend = Backend.get_backend

chat = Blueprint('chat', __name__)

@chat.route('/_connect/', methods=['GET'])
def connect():
    backend = get_backend()
    backend.connect(userid())
    return jsonify(success=True)


@chat.route('/_disconnect/', methods=['GET'])
def disconnect():
    backend = get_backend()
    backend.disconnect(userid())
    return jsonify(success=True)


@chat.route('/_check_chat_valid/', methods=['GET'])
def check_chat_valid():
    backend = get_backend()
    if backend.is_chat_valid(userid()):
        return jsonify(valid=True)
    else:
        return jsonify(valid=False, message=backend.get_user_message(userid()))

@chat.route('/_submit_survey/', methods=['POST'])
def submit_survey():
    backend = get_backend()
    data = request.json['response']
    uid = request.json['uid']
    backend.submit_survey(uid, data)
    return jsonify(success=True)

@chat.route('/_check_inbox/', methods=['GET'])
def check_inbox():
    backend = get_backend()
    uid = userid()
    event = backend.receive(uid)
    if event is not None:
        data = backend.display_received_event(event)
        return jsonify(received=True, timestamp=event.time, **data)
    else:
        return jsonify(received=False)


@chat.route('/_typing_event/', methods=['GET'])
def typing_event():
    backend = get_backend()
    action = request.args.get('action')

    uid = userid()
    chat_info = backend.get_chat_info(uid)
    backend.send(uid,
                 Event.TypingEvent(chat_info.agent_index,
                                   action,
                                   str(time.time())))

    return jsonify(success=True)


@chat.route('/_send_message/', methods=['GET'])
def send_message():
    backend = get_backend()
    message = str(request.args.get('message'))
    displayed_message = format_message(u"You: {}".format(message), False)
    uid = userid()
    time_taken = float(request.args.get('time_taken'))
    received_time = time.time()
    start_time = received_time - time_taken
    chat_info = backend.get_chat_info(uid)
    backend.send(uid,
                 Event.MessageEvent(chat_info.agent_index,
                                    message,
                                    str(received_time),
                                    str(start_time))
                 )
    return jsonify(message=displayed_message, timestamp=str(received_time))

@chat.route('/_send_eval/', methods=['POST'])
def send_eval():
    backend = get_backend()
    labels = request.json['labels']
    eval_data = request.json['eval_data']
    uid = request.json['uid']
    chat_info = backend.get_chat_info(uid)
    data = {'utterance': eval_data['utterance'], 'labels': labels}
    backend.send(uid,
                 Event.EvalEvent(chat_info.agent_index,
                                    data,
                                    eval_data['timestamp'])
                 )
    return jsonify(success=True)

@chat.route('/_join_chat/', methods=['GET'])
def join_chat():
    """Sent by clients when they enter a room.
    A status message is broadcast to all people in the room."""
    backend = get_backend()
    uid = userid()
    chat_info = backend.get_chat_info(uid)
    backend.send(uid, Event.JoinEvent(chat_info.agent_index,
                                      uid,
                                      str(time.time())))
    return jsonify(message=format_message("You entered the room.", True))


@chat.route('/_leave_chat/', methods=['GET'])
def leave_chat():
    backend = get_backend()
    uid = userid()
    chat_info = backend.get_chat_info(uid)
    backend.send(uid, Event.LeaveEvent(chat_info.agent_index,
                                       uid,
                                       str(time.time())))
    return jsonify(success=True)


@chat.route('/_check_status_change/', methods=['GET'])
def check_status_change():
    backend = get_backend()
    uid = userid()
    assumed_status = request.args.get('assumed_status')
    if backend.is_status_unchanged(uid, assumed_status):
        return jsonify(status_change=False)
    else:
        return jsonify(status_change=True)

@chat.route('/index', methods=['GET', 'POST'])
@chat.route('/', methods=['GET', 'POST'])
def index():
    """Chat room. The user's name and room must be stored in
    the session."""

    if request.args.get('assignmentId',default=None, type=None) == 'ASSIGNMENT_ID_NOT_AVAILABLE':
        return render_template('sample_chat.html',
                               kb=[{u'y': 273, u'x': 26, u'id': u'11', u'color': u'rgb(203,203,203)', u'size': 12}, {u'y': 279, u'x': 65, u'id': u'33', u'color': u'rgb(77,77,77)', u'size': 12}, {u'y': 176, u'x': 195, u'id': u'38', u'color': u'rgb(160,160,160)', u'size': 8}, {u'y': 121, u'x': 308, u'id': u'48', u'color': u'rgb(84,84,84)', u'size': 11}, {u'y': 333, u'x': 335, u'id': u'55', u'color': u'rgb(107,107,107)', u'size': 9}, {u'y': 207, u'x': 187, u'id': u'70', u'color': u'rgb(129,129,129)', u'size': 11}, {u'y': 203, u'x': 56, u'id': u'74', u'color': u'rgb(127,127,127)', u'size': 8}],
                               title=app.config['task_title'],
                               instructions=Markup(app.config['instructions']),
                               icon=app.config['task_icon'])

    if not request.args.get('uid'):
        prefix = "U_"
        if request.args.get('mturk') and int(request.args.get('mturk')) == 1:
            # link for Turkers
            prefix = "MT_"
        elif request.args.get('nlp') and int(request.args.get('nlp')) == 1:
            # link for NLP group
            prefix = "NLP_"

        url = request.url + '?{}={}'.format('uid', generate_userid(prefix))
        for _key in dict(request.args).keys():
            _value = request.args.get(_key,default=None, type=None)
            if _value:
                url += '&{}={}'.format(_key, _value)
        return redirect(url)

    backend = get_backend()

    backend.create_user_if_not_exists(userid())

    status = backend.get_updated_status(userid())

    # request args
    hitId = request.args.get('hitId', default=None, type=None)
    assignmentId = request.args.get('assignmentId', default=None, type=None)
    turkSubmitTo = request.args.get('turkSubmitTo', default=None, type=None)
    workerId = request.args.get('workerId', default=None, type=None)

    mturk = True if hitId else None

    if status == Status.Waiting:
        waiting_info = backend.get_waiting_info(userid())
        return render_template('waiting.html',
                               seconds_until_expiration=waiting_info.num_seconds,
                               waiting_message=waiting_info.message,
                               uid=userid(),
                               title=app.config['task_title'],
                               icon=app.config['task_icon'])
    elif status == Status.Finished:
        finished_info, chat_id, completed = backend.get_finished_info(userid(), from_mturk=mturk, hit_id=hitId, assignment_id=assignmentId)
        if completed:
          backend.add_completed_scenarios(chat_id, userid())
        visualize_link = False
        if request.args.get('debug') is not None and request.args.get('debug') == '1':
            visualize_link = True
        return render_template('finished.html',
                               finished_message=finished_info.message,
                               chat_id=chat_id,
                               turk_submit_to=request.args.get('turkSubmitTo', default=None, type=None),
                               title=app.config['task_title'],
                               icon=app.config['task_icon'],
                               visualize=visualize_link,
                               uid=userid(),
                               completed=completed)
    elif status == Status.Incomplete:
        finished_info, chat_id, completed = backend.get_finished_info(userid(), from_mturk=False, current_status=Status.Incomplete)
        visualize_link = False
        if request.args.get('debug') is not None and request.args.get('debug') == '1':
            visualize_link = True
        return render_template('finished.html',
                               finished_message=finished_info.message,
                               chat_id=chat_id,
                               turk_submit_to=request.args.get('turkSubmitTo', default=None, type=None),
                               title=app.config['task_title'],
                               icon=app.config['task_icon'],
                               visualize=visualize_link,
                               uid=userid(),
                               completed=False)
    elif status == Status.Chat:
        debug = False
        peek = False
        partner_kb = None
        if request.args.get('debug') is not None and request.args.get('debug') == '1':
            debug = True
        if request.args.get('peek') is not None and request.args.get('peek') == '1':
            peek = True
        chat_info = backend.get_chat_info(userid(), peek)
        if peek:
            partner_kb = chat_info.partner_kb.to_dict()
        return render_template('chat.html',
                               debug=debug,
                               uid=userid(),
                               kb=chat_info.kb.to_dict(),
                               attributes=[attr.name for attr in chat_info.attributes],
                               num_seconds=chat_info.num_seconds,
                               title=app.config['task_title'],
                               instructions=Markup(app.config['instructions']),
                               icon=app.config['task_icon'],
                               partner_kb=partner_kb,
                               quit_enabled=app.config['user_params']['skip_chat_enabled'],
                               quit_after=app.config['user_params']['status_params']['chat']['num_seconds'] -
                                          app.config['user_params']['quit_after'])
    elif status == Status.Survey:
        survey_info = backend.get_survey_info(userid())
        visualization = None
        return render_template('task_survey.html',
                               title=app.config['task_title'],
                               uid=userid(),
                               icon=app.config['task_icon'],
                               kb=survey_info.kb.to_dict(),
                               partner_kb=survey_info.partner_kb.to_dict(),
                               attributes=[attr.name for attr in survey_info.attributes],
                               message=survey_info.message,
                               results=survey_info.result,
                               agent_idx=survey_info.agent_idx,
                               scenario_id=survey_info.scenario_id,
                               visualization=visualization)
    elif status == Status.Reporting:
        return render_template('report.html',
                               title=app.config['task_title'],
                               uid=userid(),
                               icon=app.config['task_icon'])

@chat.route('/_report/', methods=['GET'])
def report():
    backend = get_backend()
    uid = userid()
    feedback = request.args.get('feedback')
    backend.report(uid, feedback)
    return jsonify(success=True)


@chat.route('/_init_report/', methods=['GET'])
def init_report():
    backend = get_backend()
    uid = userid()
    backend.init_report(uid)
    return jsonify(success=True)
