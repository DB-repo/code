from flask import Blueprint
from flask import flash
from flask import g
from flask import redirect
from flask import render_template
from flask import request
from flask import url_for
from flask import json, Response
from werkzeug.exceptions import abort

import threading
import time

from db import get_db
from process import Query, Baseline1Query, ProgressiveBaselinesQuery, Results, Plan, Explain, State, Schema, Performance
from approximation import WiFiApproximation, ExplainWiFiApproximation
from restart import pause

bp = Blueprint("blog", __name__)

query_thread = {}
query_response = {}
query_plan = {}

approx_query = None


ERROR = {
    "error": 1
}

SUCCESS = {
    "success": 1
}


@bp.route("/execute", methods=("POST",))
def execute():
    """Create a new post for the current user."""

    global query_thread
    global query_response
    global query_plan

    payload = request.json
    query = payload['query']
    delay = int(payload['delay'])
    epochs = int(payload['epochs'])
    group = payload['group']
    token = payload['token']

    if token in query_thread:
        return Response(json.dumps(ERROR), status=415, mimetype='application/json')

    if not request.headers['Content-Type'] == 'application/json':
        return Response(json.dumps(ERROR), status=415, mimetype='application/json')

    query_thread[token] = Query(query, delay, epochs, group, 1, token)
    query_thread[token].start()

    query_response[token] = Results(group, token)
    query_plan[token] = Plan(token)

    time.sleep(2)

    if query_thread[token].exception is not None:
        return Response(json.dumps(ERROR), status=415, mimetype='application/json')

    return Response(json.dumps(SUCCESS), status=200, mimetype='application/json')


@bp.route("/baseline_execute", methods=("POST",))
def baseline_execute():
    """Create a new post for the current user."""

    global in_progress
    global query_thread
    global query_response
    global query_plan

    payload = request.json
    query = payload['query']
    delay = int(payload['delay'])
    epochs = int(payload['epochs'])
    group = payload['group']
    baseline = payload['baseline']
    token = payload['token']

    if token in query_thread:
        return Response(json.dumps(ERROR), status=415, mimetype='application/json')

    if not request.headers['Content-Type'] == 'application/json':
        return Response(json.dumps(ERROR), status=415, mimetype='application/json')

    query_thread[token] = ProgressiveBaselinesQuery(query, delay, epochs, group, 1, baseline, token)
    query_thread[token].start()

    query_response[token] = Results(group, token)
    query_plan[token] = Plan(token)
    in_progress = True

    time.sleep(2)

    if query_thread[token].exception is not None:
        return Response(json.dumps(ERROR), status=415, mimetype='application/json')

    return Response(json.dumps(SUCCESS), status=200, mimetype='application/json')


@bp.route("/npexecute", methods=("POST",))
def npexecute():
    """Create a new post for the current user."""

    if not request.headers['Content-Type'] == 'application/json':
        return Response(json.dumps(ERROR), status=415, mimetype='application/json')

    payload = request.json
    query = payload['query']

    query_exec = Baseline1Query(query)

    return Response(json.dumps(query_exec.run()), status=200, mimetype='application/json')


@bp.route("/stop", methods=("POST",))
def stop():
    """Create a new post for the current user."""
    global query_thread
    global query_response
    global query_plan

    payload = request.json
    token = payload['token']

    if token not in query_thread:
        return Response(json.dumps(ERROR), status=415, mimetype='application/json')
    else:
        if query_thread[token].is_alive():
            query_thread[token].close()

        del query_thread[token]
        del query_response[token]
        del query_plan[token]

    return Response(json.dumps(SUCCESS), status=200, mimetype='application/json')


@bp.route("/pause", methods=("POST",))
def pause_query():
    """Create a new post for the current user."""
    global query_thread
    global query_response
    global query_plan

    payload = request.json
    token = payload['token']

    if token not in query_thread:
        return Response(json.dumps(ERROR), status=415, mimetype='application/json')
    else:
        pause(token)

        del query_thread[token]
        del query_response[token]
        del query_plan[token]

    return Response(json.dumps(SUCCESS), status=200, mimetype='application/json')


@bp.route("/reload", methods=("GET", "POST"))
def reload_results():
    """Update a post if the current user is the author."""
    global query_thread
    global query_response

    payload = request.json
    token = payload['token']

    if token not in query_thread:
        if approx_query is not None and token in approx_query.queries:
            return Response(json.dumps([approx_query.results[token].get_columns(), approx_query.results[token].fetch()]), status=200, mimetype='application/json')
        else:
            return Response(json.dumps(ERROR), status=415, mimetype='application/json')
    else:
        return Response(json.dumps([query_response[token].get_columns(), query_response[token].fetch()]), status=200, mimetype='application/json')


@bp.route("/plan", methods=("GET", "POST"))
def fetch_plan():
    """Update a post if the current user is the author."""
    global query_thread
    global query_response
    global query_plan

    payload = request.json
    token = payload['token']

    if token not in query_thread:
        if approx_query is not None and token in approx_query.queries:
            plan = Plan(token)
            return Response(json.dumps(plan.fetch_epoch_plan()), status=200, mimetype='application/json')
        else:
            return Response(json.dumps(ERROR), status=415, mimetype='application/json')
    else:
        return Response(json.dumps(query_plan[token].fetch_epoch_plan()), status=200, mimetype='application/json')


@bp.route("/explain", methods=("GET", "POST"))
def fetch_explain():
    """Update a post if the current user is the author."""
    if not request.headers['Content-Type'] == 'application/json':
        return Response(json.dumps(ERROR), status=415, mimetype='application/json')

    payload = request.json
    query = payload['query']
    algo = payload['algo']

    explain = Explain(query, algo)

    return Response(json.dumps({"explain":explain.fetch_explain()}), status=200, mimetype='application/json')


@bp.route("/state", methods=("GET", "POST"))
def fetch_state():
    """Update a post if the current user is the author."""
    if not request.headers['Content-Type'] == 'application/json':
        return Response(json.dumps(ERROR), status=415, mimetype='application/json')

    payload = request.json
    type_ = payload['type']
    oid = payload['id']
    state = State(type_, oid)

    return Response(json.dumps(state.fetch_state()), status=200, mimetype='application/json')


@bp.route("/schema", methods=("GET", "POST"))
def fetch_schema():
    """Update a post if the current user is the author."""
    if not request.headers['Content-Type'] == 'application/json':
        return Response(json.dumps(ERROR), status=415, mimetype='application/json')

    schema = Schema()
    result = {
        "attributes": schema.fetch_tables_and_attrs(),
        "classes": schema.fetch_function_classes(),
        "functions": schema.fetch_function_table(),
        "decisions": schema.fetch_decision_table()
    }

    return Response(json.dumps(result), status=200, mimetype='application/json')


@bp.route("/perf", methods=("GET", "POST"))
def fetch_performance():
    """Update a post if the current user is the author."""
    global query_thread
    global query_response
    global query_plan

    payload = request.json
    token = payload['token']

    if token not in query_thread:
        if approx_query is not None and token in approx_query.queries:
            perf = Performance(token)
            return Response(json.dumps(perf.fetch_perf()), status=200, mimetype='application/json')
        else:
            return Response(json.dumps(ERROR), status=415, mimetype='application/json')
    else:
        perf = Performance(token)
        return Response(json.dumps(perf.fetch_perf()), status=200, mimetype='application/json')


@bp.route("/approx_execute", methods=("POST",))
def approx_execute():
    global approx_query

    payload = request.json
    query = payload['query']
    delay = int(payload['delay'])
    epochs = int(payload['epochs'])
    group = payload['group']

    if approx_query is not None:
        return Response(json.dumps(ERROR), status=415, mimetype='application/json')

    if not request.headers['Content-Type'] == 'application/json':
        return Response(json.dumps(ERROR), status=415, mimetype='application/json')

    approx_query = WiFiApproximation(query, delay, epochs, group)
    approx_query.start()
    time.sleep(2)

    return Response(json.dumps(SUCCESS), status=200, mimetype='application/json')


@bp.route("/approx_reload", methods=("POST",))
def approx_reload():
    global approx_query

    if approx_query is None:
        return Response(json.dumps(ERROR), status=415, mimetype='application/json')
    else:
        return Response(json.dumps(approx_query.reload()), status=200, mimetype='application/json')


@bp.route("/approx_plan", methods=("POST",))
def approx_plan():
    global approx_query

    if approx_query is None:
        return Response(json.dumps(ERROR), status=415, mimetype='application/json')
    else:
        return Response(json.dumps(approx_query.plan()), status=200, mimetype='application/json')


@bp.route("/approx_stop", methods=("POST",))
def approx_stop():
    global approx_query

    if approx_query is None:
        return Response(json.dumps(ERROR), status=415, mimetype='application/json')
    else:
        approx_query.stop()
        approx_query = None
    return Response(json.dumps(SUCCESS), status=200, mimetype='application/json')

@bp.route("/approx_explain", methods=("POST",))
def approx_explain():

    if not request.headers['Content-Type'] == 'application/json':
        return Response(json.dumps(ERROR), status=415, mimetype='application/json')

    payload = request.json
    query = payload['query']
    explain = ExplainWiFiApproximation(query)

    return Response(json.dumps({"explain": explain.explain()}), status=200, mimetype='application/json')

