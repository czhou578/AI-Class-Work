'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import copy
import queue


def standardize_variables(nonstandard_rules):
    '''
    @param nonstandard_rules (dict) - dict from ruleIDs to rules
        Each rule is a dict:
        rule['antecedents'] contains the rule antecedents (a list of propositions)
        rule['consequent'] contains the rule consequent (a proposition).

    @return standardized_rules (dict) - an exact copy of nonstandard_rules,
        except that the antecedents and consequent of every rule have been changed
        to replace the word "something" with some variable name that is
        unique to the rule, and not shared by any other rule.
    @return variables (list) - a list of the variable names that were created.
        This list should contain only the variables that were used in rules.
    '''
    counter = 0
    variables = []
    copy_of_input = copy.deepcopy(nonstandard_rules)

    for rule, ruleValues in copy_of_input.items():
        for x in ruleValues['antecedents']:
            if (len(x) > 1):
                x[0] = "rule" + str(counter)
                ruleValues['consequent'][0] = "rule" + str(counter)
                variables.append("rule" + str(counter))
                counter += 1
            break

    return copy_of_input, variables


def unify(query, datum, variables):
    '''
    @param query: proposition that you're trying to match.
      The input query should not be modified by this function; consider deepcopy.
    @param datum: proposition against which you're trying to match the query.
      The input datum should not be modified by this function; consider deepcopy.
    @param variables: list of strings that should be considered variables.
      All other strings should be considered constants.

    Unification succeeds if (1) every variable x in the unified query is replaced by a 
    variable or constant from datum, which we call subs[x], and (2) for any variable y
    in datum that matches to a constant in query, which we call subs[y], then every 
    instance of y in the unified query should be replaced by subs[y].

    @return unification (list): unified query, or None if unification fails.
    @return subs (dict): mapping from variables to values, or None if unification fails.
       If unification is possible, then answer already has all copies of x replaced by
       subs[x], thus the only reason to return subs is to help the calling function
       to update other rules so that they obey the same substitutions.

    Examples:

    GOOOOOOD
    unify(['x', 'eats', 'y', False], ['a', 'eats', 'b', False], ['x','y','a','b'])
      unification = [ 'a', 'eats', 'b', False ]
      subs = { "x":"a", "y":"b" }

    GOOOOOOD
    unify(['bobcat','eats','y',True],['a','eats','squirrel',True], ['x','y','a','b'])
      unification = ['bobcat','eats','squirrel',True]
      subs = { 'a':'bobcat', 'y':'squirrel' }

    unify(['x','eats','x',True],['a','eats','a',True],['x','y','a','b'])
      unification = ['a','eats','a',True]
      subs = { 'x':'a' }

    unify(['x','eats','x',True],['a','eats','bobcat',True],['x','y','a','b'])
      unification = ['bobcat','eats','bobcat',True],

      subs = {'x':'a', 'a':'bobcat'}

      When the 'x':'a' substitution is detected, the query is changed to 
      ['a','eats','a',True].  Then, later, when the 'a':'bobcat' substitution is 
      detected, the query is changed to ['bobcat','eats','bobcat',True], which 
      is the value returned as the answer.

    unify(['a','eats','bobcat',True],['x','eats','x',True],['x','y','a','b'])
      unification = ['bobcat','eats','bobcat',True],
      subs = {'a':'x', 'x':'bobcat'}
      When the 'a':'x' substitution is detected, the query is changed to 
      ['x','eats','bobcat',True].  Then, later, when the 'x':'bobcat' substitution 
      is detected, the query is changed to ['bobcat','eats','bobcat',True], which is 
      the value returned as the answer.

    unify([...,True],[...,False],[...]) should always return None, None, regardless of the 
      rest of the contents of the query or datum.
    '''

    copy_of_query = copy.deepcopy(query)
    copy_of_datum = copy.deepcopy(datum)
    track_query_var_index = []  # tracking indices of variables in query

    unification = []
    subs = {}

    if (copy_of_query[len(copy_of_query) - 1] != copy_of_datum[len(copy_of_datum) - 1]):
        return None, None

    for x in range(len(copy_of_datum) - 1):

        if (len(copy_of_datum[x]) == 1 and len(copy_of_query[x]) != 1 and copy_of_datum[x] in variables):
            curr_element = copy_of_query[x]

            if (copy_of_datum[x] in subs.values() or copy_of_datum[x] in subs.keys()):
                for x in track_query_var_index:
                    unification[x] = curr_element

            unification.append(curr_element)
            subs[copy_of_datum[x]] = curr_element
            track_query_var_index.append(x)

        elif (len(copy_of_datum[x]) != 1 and len(copy_of_query[x]) == 1):
            curr_element = copy_of_datum[x]

            if (copy_of_query[x] in subs):
                for x in track_query_var_index:
                    unification[x] = curr_element

                value = subs[copy_of_query[x]]
                subs[value] = curr_element
                unification.append(curr_element)

            else:
                unification.append(curr_element)

                subs[copy_of_query[x]] = curr_element

            track_query_var_index.append(x)

        elif (len(copy_of_datum[x]) == 1 and len(copy_of_query[x]) == 1):
            unification.append(copy_of_datum[x])
            subs[copy_of_query[x]] = copy_of_datum[x]
            track_query_var_index.append(x)

        elif (len(copy_of_datum[x]) != 1 and len(copy_of_query[x]) != 1):
            if (any(variable in copy_of_query[x] for variable in variables) and len(copy_of_datum[x]) != len(copy_of_query[x])):
                unification.append(copy_of_datum[x])
                subs[copy_of_query[x]] = copy_of_datum[x]
                track_query_var_index.append(x)
            else:
                unification.append(copy_of_datum[x])

    unification.append(copy_of_datum[len(copy_of_query) - 1])

    return unification, subs


def apply(rule, goals, variables):
    '''
    @param rule: A rule that is being tested to see if it can be applied
      This function should not modify rule; consider deepcopy.
    @param goals: A list of propositions against which the rule's consequent will be tested
      This function should not modify goals; consider deepcopy.
    @param variables: list of strings that should be treated as variables

    Rule application succeeds if the rule's consequent can be unified with any one of the goals.

    @return applications: a list, possibly empty, of the rule applications that
       are possible against the present set of goals.
       Each rule application is a copy of the rule, but with both the antecedents 
       and the consequent modified using the variable substitutions that were
       necessary to unify it to one of the goals. Note that this might require 
       multiple sequential substitutions, e.g., converting ('x','eats','squirrel',False)
       based on subs=={'x':'a', 'a':'bobcat'} yields ('bobcat','eats','squirrel',False).
       The length of the applications list is 0 <= len(applications) <= len(goals).  
       If every one of the goals can be unified with the rule consequent, then 
       len(applications)==len(goals); if none of them can, then len(applications)=0.
    @return goalsets: a list of lists of new goals, where len(newgoals)==len(applications).
       goalsets[i] is a copy of goals (a list) in which the goal that unified with 
       applications[i]['consequent'] has been removed, and replaced by 
       the members of applications[0]['antecedents'].

    Example:
    rule={
      'antecedents':[['x','is','nice',True],['x','is','hungry',False]],
      'consequent':['x','eats','squirrel',False]
    }
    goals=[
      ['bobcat','eats','squirrel',False],
      ['bobcat','visits','squirrel',True],
      ['bald eagle','eats','squirrel',False]
    ]
    variables=['x','y','a','b']

    applications, newgoals = submitted.apply(rule, goals, variables)

    applications==[
      {
        'antecedents':[['bobcat','is','nice',True],['bobcat','is','hungry',False]],
        'consequent':['bobcat','eats','squirrel',False]
      },
      {
        'antecedents':[['bald eagle','is','nice',True],['bald eagle','is','hungry',False]],
        'consequent':['bald eagle','eats','squirrel',False]
      }
    ]
    newgoals==[
      [
        ['bobcat','visits','squirrel',True], from goals
        ['bald eagle','eats','squirrel',False] from goals
        ['bobcat','is','nice',True], from applications['antecedents'][0]
        ['bobcat','is','hungry',False] from applications['antecedents'][1]
      ],[
        ['bobcat','eats','squirrel',False], from goals
        ['bobcat','visits','squirrel',True], from goals
        ['bald eagle','is','nice',True], applications['antecedents']
        ['bald eagle','is','hungry',False], applications['antecedents]
      ]
    '''
    copy_rule = copy.deepcopy(rule)
    applications = []
    goalsets = []

    for entry in goals:
        applications_entry = {}
        conseq_unify, subs = unify(copy_rule['consequent'], entry, variables)

        if (conseq_unify == None and subs == None):
            continue
        else:
            applications_entry['consequent'] = conseq_unify
            applications_entry['antecedents'] = []

            for anteced in copy_rule['antecedents']:
                copy_anteced = copy.deepcopy(anteced)

                for item in range(len(anteced) - 1):
                    if (anteced[item] in subs):
                        copy_anteced[item] = subs[anteced[item]]
                applications_entry['antecedents'].append(copy_anteced)

            applications.append(applications_entry)

    for entry in applications:
        goalset_entry = []
        copy_goals = copy.deepcopy(goals)
        entry_consequent = entry['consequent']
        entry_antecedents = entry['antecedents']

        for entry in copy_goals:
            if (all(x == y for x, y in zip(entry, entry_consequent))):
                continue
            goalset_entry.append(entry)

        for anteced in entry_antecedents:
            goalset_entry.append(anteced)

        goalsets.append(goalset_entry)

    return applications, goalsets


def backward_chain(query, rules, variables):
    '''
    @param query: a proposition, you want to know if it is true
    @param rules: dict mapping from ruleIDs to rules

    @return proof (list): a list of rule applications
      that, when read in sequence, conclude by proving the truth of the query.
      If no proof of the query was found, you should return proof=None.
    '''

    goalset_queue = queue.Queue()
    goalset_queue.put({'rule_apps': [], 'query': [query]})

    while not goalset_queue.empty():

        front_goalset_queue = goalset_queue.get()
        if (front_goalset_queue['query'] == [[]]):
            return front_goalset_queue['rule_apps']

        for rule in rules.keys():
            if ('triple' in rule and rules[rule]['consequent'] in front_goalset_queue['query']):
                front_goalset_queue['query'].remove(rules[rule]['consequent'])
                front_goalset_queue['rule_apps'].append(rule)

                if (len(front_goalset_queue['query']) == 0):
                    return front_goalset_queue['rule_apps']

                continue
            else:
                applications, goalsets = apply(
                    {'antecedents': rules[rule]['antecedents'], 'consequent': rules[rule]['consequent']}, front_goalset_queue['query'], variables)

                if (len(applications) == 0):
                    continue

                new_rule_apps_copy = front_goalset_queue['rule_apps'].copy()
                new_rule_apps_copy.append(rule)

                if (goalsets == [[]]):
                    new_queue_entry = {
                        'rule_apps': new_rule_apps_copy, 'query': goalsets}
                    goalset_queue.put(new_queue_entry)

                for goal in goalsets:
                    new_queue_entry = {
                        'rule_apps': new_rule_apps_copy, 'query': goal}
                    goalset_queue.put(new_queue_entry)

    return None
