from flask import Blueprint, jsonify, request
from cocoa.web.views.utils import userid, format_message
from web.main.backend import get_backend
import pdb

action = Blueprint('action', __name__)

@action.route('/_select_option/', methods=['GET'])
def select_option():
    backend = get_backend()
    selection_id = request.args.get('selection')
    selected_item = backend.select(userid(), selection_id)
    displayed_message = format_message("You selected", True)
    return jsonify(message=displayed_message)
