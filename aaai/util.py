import dateutil

from kgtk.functions import kgtk

default_columns = ['id', 'node1', 'label', 'node2', 'lang', 'rank', 'node2;wikidatatype']

PROPERTY_COLUMNS = ['id', 'label', 'description', 'node2;wikidatatype']
NODE_COLUMNS = ['id', 'label', 'description', 'alias']
CLASS_NODE_COLUMNS = ['id', 'label', 'description', 'alias', 'Psubclass_of']
INSTANCE_NODE_COLUMNS = ['id', 'label', 'description', 'alias', 'Pisa']
ALL_NODE_COLUMNS = ['id', 'label', 'description', 'alias', 'Psubclass_of', 'Pisa']
BROWSER_DEFAULT_COLUMNS = ['id', 'node1', 'label', 'node2', 'lang', 'rank', 'node2;wikidatatype']



def to_second_precision(literal):
    '''Return ISO time string truncated to second precision'''
    first = literal.split('.')[0]
    return f'{first}/14'


def to_hour_precision(literal):
    '''Return ISO time string truncated to hour precision'''
    first = literal.split(':')[0] + ':00:00'
    return f'{first}/12'


def to_date(literal) -> str:
    if literal[0] == '^':
        return literal[1:-3]
    return literal


def to_datetime(literal) -> "datetime.datetime":
    return dateutil.parser.isoparse(to_date(literal))

def add_time_block_column(normalized_kgtk_df, kgtk_df, *, time_column='time'):
    block_idx_map = {}
    for id, row in kgtk_df.iterrows():
        block_idx_map[row['id']] = to_second_precision(row['time'])
    normalized_kgtk_df.insert(0, 'block', normalized_kgtk_df['node1'].apply(
        lambda x: block_idx_map[x]))
    return normalized_kgtk_df

def en(x):
    if x.endswith('@en'):
        return x
    else:
        return f"'{x}'@en"

def drop_en(x):
    if isinstance(x, str) and x.endswith('@en'):
        return x[1:-4]
    else:
        return x

def drop_precision(x):
    if x.startswith('^'):
        return x[1:-3]
    return x

def to_labels(node_df):
    result = node_df[node_df['label'].notna()].apply(
        lambda node:
            [f'{node.id}-label-en', node.id, 'label', node.label, 'en', '', ''],
        axis=1, result_type='expand')
    result.columns = default_columns
    return result

def to_alias(node_df):
    alias = node_df.loc[:, ['id', 'alias']]
    alias = alias[alias['alias'].notna()]
    result = kgtk(alias, 'normalize-nodes / add-id --id-style wikidata')
    result['lang'] = 'en'
    result['rank'] = ''
    result['node2;wikidatatype'] = ''
    result = result.loc[:, default_columns]
    return result

def to_desc(node_df):
    desc = node_df.loc[:, ['id', 'description']]
    desc = desc[desc['description'].notna()]
    result = desc.apply(
        lambda node:
            [f'{node.id}-label-en', node.id, 'description', node.description, 'en', '', ''],
        axis=1, result_type='expand')
    result.columns = default_columns
    return result

def find_datatype(row):
    if row.label == 'Pcontains_edge':
        return 'external-id'
    if type(row.node2) == str:
        if row.node2[0] == '^':
            return 'time'
        elif row.node2[0] == 'Q':
            return 'wikibase-item'
        elif row.node2[0] in '0123456789-':
            return 'quantity'
        else:
            return 'string'
    else:
        return 'quantity'

def to_claims(edge_df):
    result = edge_df[~edge_df['label'].isin(['label', 'alias', 'description'])].loc[:, ['id', 'node1', 'label', 'node2']]
    result['lang'] = ''
    result['rank'] = ''
    result['node2;wikidatatype'] = result.apply(find_datatype, axis=1)
    return result

def to_claims_item(claims_df):
    return claims_df[claims_df['node2;wikidatatype']=='wikibase-item']

def to_qualifiers(edge_df):
    result = kgtk(edge_df.loc[:, ['id', 'Ptime', 'Pgraph']], 'normalize-nodes / add-id --id-style wikidata')
    result['lang'] = ''
    result['rank'] = ''
    result['node2;wikidatatype'] = result.apply(find_datatype, axis=1)
    result = result.loc[:, default_columns]
    return result

def to_property_datatype(node_df):
    prop = node_df[node_df['node2;wikidatatype'].notna()]
    prop = prop.apply(lambda node: [f'{node.id}-datatype', node.id, 'datatype', node['node2;wikidatatype'], '', '', '' ], axis=1, result_type='expand')
    prop.columns = default_columns
    return prop
