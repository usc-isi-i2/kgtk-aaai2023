from collections import defaultdict
from operator import itemgetter
import bisect
import csv
import datetime
import math
from typing import Dict

import colorsys
import numpy as np
import pandas as pd

from kgtk.functions import kgtk, kypher

from .util import en, drop_en, CLASS_NODE_COLUMNS, PROPERTY_COLUMNS, BROWSER_DEFAULT_COLUMNS, NODE_COLUMNS, to_datetime, drop_precision

# from fibo import _QNODES, _PROPERTY_NODES
from .ontology import _QNODES, _PROPERTY_NODES


# [['P31', en('instance of'),
#   en('that class of which this subject is a particular example and member'),
#   'wikibase-item'],
#  ['P585', en('point in time'),
#   en('time and date something took place, existed or a statement was true'),
#   'time'],
#  ['P279', en('subclass of'), en('next higher class or type; all instances of these items are instances of those items; this item is a class (subset) of that item. ')]]

class Transaction:
    class Qnode:
        @classmethod
        def get_dataframe(cls):
            return _QNODES.loc[:, CLASS_NODE_COLUMNS]

        def __getattr__(self, abbrev):
            result = _QNODES[_QNODES['abbrev'] == abbrev]
            if result.size == 0:
                raise Exception(f'Qnode not found: {abbrev}')
            return result['id'].iloc[0]


    class Property:
        @classmethod
        def get_dataframe(cls):
            return _PROPERTY_NODES.loc[:, PROPERTY_COLUMNS]

        def __getattr__(self, abbrev):
            result = _PROPERTY_NODES[_PROPERTY_NODES['abbrev'] == abbrev]
            if result.size == 0:
                raise Exception(f'Property not found: {abbrev}')
            return result['id'].iloc[0]

    # class Edge:
    #     @classmethod
    #     def get_dataframe(cls):
    #         return subclass_edges

    def __init__(self):
        self.q = self.Qnode()
        self.p = self.Property()


    def get_org_nodes(self, node_df, names_file: str = 'species.txt'):
        '''Returns organization nodes'''
        orgs = list(set(node_df[self.p.buyer].values).union(set(node_df[self.p.seller].values)))
        orgs.sort()
        org_list = []
        with open(names_file) as fd:
            names = fd.readlines()
        for org, name in zip(orgs, names):
            org_num = org.split('_')[-1]
            org_list.append(
                [org, en(f'{name.strip()} ({org_num})'), en('Organization ' + org_num),
                 en('org_' + org_num), self.q.org])
        org_nodes = pd.DataFrame(org_list, columns=NODE_COLUMNS + [self.p.isa])
        return org_nodes


    def to_labels(self, node_df):
        result = node_df[node_df['label'].notna()].apply(
            lambda node:
                [f'{node.id}-label-en', node.id, 'label', node.label, 'en', '', ''],
            axis=1, result_type='expand')
        result.columns = BROWSER_DEFAULT_COLUMNS
        return result

    def to_alias(self, node_df):
        alias = node_df.loc[:, ['id', 'alias']]
        alias = alias[alias['alias'].notna()]
        result = kgtk(alias, 'normalize-nodes / add-id --id-style wikidata')
        result['lang'] = 'en'
        result['rank'] = ''
        result['node2;wikidatatype'] = ''
        result = result.loc[:, BROWSER_DEFAULT_COLUMNS]
        return result

    def to_desc(self, node_df):
        desc = node_df.loc[:, ['id', 'description']]
        desc = desc[desc['description'].notna()]
        result = desc.apply(
            lambda node:
                [f'{node.id}-label-en', node.id, 'description', node.description, 'en', '', ''],
            axis=1, result_type='expand')
        result.columns = BROWSER_DEFAULT_COLUMNS
        return result

    def find_datatype(self, row):
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

    def to_node_claims(self, node_df):
        partial = node_df.drop(columns=['label', 'description', 'alias'])
        result = kgtk(partial, "normalize-nodes / add-id --id-style wikidata").dropna()
        result = result.loc[:, ['id', 'node1', 'label', 'node2']]
        result['lang'] = ''
        result['rank'] = ''
        result['node2;wikidatatype'] = result.apply(self.find_datatype, axis=1)
        return result


    def to_claims(self, edge_df):
        result = edge_df[~edge_df['label'].isin(['label', 'alias', 'description'])].loc[:, ['id', 'node1', 'label', 'node2']]
        result['lang'] = ''
        result['rank'] = ''
        result['node2;wikidatatype'] = result.apply(self.find_datatype, axis=1)
        return result

    def to_claims_item(self, claims_df):
        return claims_df[claims_df['node2;wikidatatype']=='wikibase-item']

    def to_qualifiers(self, edge_df):
        result = kgtk(edge_df.loc[:, ['id', self.p.time, self.p.graph]], 'normalize-nodes / add-id --id-style wikidata')
        result['lang'] = ''
        result['rank'] = ''
        result['node2;wikidatatype'] = result.apply(self.find_datatype, axis=1)
        result = result.loc[:, BROWSER_DEFAULT_COLUMNS]
        return result

    def to_property_datatype(self, node_df):
        prop = node_df[node_df['node2;wikidatatype'].notna()]
        prop = prop.apply(lambda node: [f'{node.id}-datatype', node.id, 'datatype', node['node2;wikidatatype'], '', '', '' ], axis=1, result_type='expand')
        prop.columns = BROWSER_DEFAULT_COLUMNS
        return prop

    def get_label_map(self, labels_tsv_file) -> Dict[str,str]:
        '''Returns Qnode to label string dict'''
        labels_df = pd.read_csv(labels_tsv_file, sep='\t')
        label_map = {}
        for row in labels_df.itertuples():
            label_map[row.node1] = drop_en(row.node2)
        return label_map

    def get_tableau_view(self, trades: pd.DataFrame, dropped: list, label_map: dict) -> pd.DataFrame:
        '''Transaction edge format to format more suitable for viewing in Tableau'''
        tableau_df = pd.DataFrame(index=trades.index)
        tableau_df['transaction'] = trades.apply(lambda x: drop_en(x['label']), axis=1)
        tableau_df['buyer'] = trades.apply(lambda x: label_map[x[self.p.buyer]], axis=1)
        tableau_df['seller'] = trades.apply(lambda x: label_map[x[self.p.seller]], axis=1)
        tableau_df['time'] = trades.apply(lambda x: drop_precision(x[self.p.time]), axis=1)
        tableau_df['asset'] = 'symbol A'
        tableau_df['price'] = trades.apply(lambda x: x[self.p.price], axis=1)
        tableau_df['quantity'] = trades.apply(lambda x: x[self.p.quantity], axis=1)
        tableau_df['dropped'] = dropped
        return tableau_df

    def get_graph_nodes(self) -> pd.DataFrame:
        buy_df = kypher(f"""
-i claims -i label
--match '
(org)-[:{self.p.isa}]->(:{self.q.org}),
(txn)-[:{self.p.buyer}]->(org),
(txn)-[:{self.p.quantity}]->(quantity),
label: (org)-[:label]->(lorg)'
--return 'org as id, sum(quantity) as quantity, count(txn) as transactions'
""")
        sell_df = kypher(f"""
-i claims -i label
--match '
(org)-[:{self.p.isa}]->(:{self.q.org}),
(txn)-[:{self.p.seller}]->(org),
(txn)-[:{self.p.quantity}]->(quantity),
label: (org)-[:label]->(lorg)'
--return 'org as id, sum(quantity) as quantity, count(txn) as transactions'
""")

        buy_df = buy_df.set_index('id')
        sell_df = sell_df.set_index('id')
        total_df = buy_df.add(sell_df,  fill_value=0)

        graph_node = kypher(f"""
-i claims -i label
--match '
(org)-[:{self.p.isa}]->(:{self.q.org}),
label: (org)-[:label]->(lorg)'
--return 'distinct org as id, lorg as label'
""")
        graph_node = graph_node.set_index('id')
        graph_node['transactions'] = total_df['transactions']
        graph_node['quantity'] = total_df['quantity']
        graph_node['tooltip'] = graph_node.apply(lambda r: '<br>'.join([drop_en(r.label), str(r.transactions), str(r.quantity)]), axis=1)
        graph_node = graph_node.reset_index()
        return graph_node

    def get_graph_edges(self, node_set=None) -> pd.DataFrame:
        buy_edges = kypher(f'''
-i claims -i label
--match '
(txn)-[:{self.p.isa}]->(:{self.q.txn}),
(txn)-[:{self.p.buyer}]->(qbuyer),
(txn)-[:{self.p.seller}]->(qseller),
(txn)-[:{self.p.quantity}]->(quantity)'
--return 'qbuyer as buyer, qseller as seller, count(txn) as transactions, sum(quantity) as quantity'
''')
        txn_map = {}
        for i, row in buy_edges.iterrows():
            if node_set and not (row.seller in node_set and row.buyer in node_set):
                continue
            if (row.seller, row.buyer) in txn_map:
                txn, quant = txn_map[(row.seller, row.buyer)]
                txn_map[(row.seller, row.buyer)] = (txn + row.transactions, quant + row.quantity)
            else:
                txn_map[(row.seller, row.buyer)] = (row.transactions, row.quantity)
            if (row.buyer, row.seller) in txn_map:
                txn, quant = txn_map[(row.buyer, row.seller)]
                txn_map[(row.buyer, row.seller)] = (txn + row.transactions, quant + row.quantity)
            else:
                txn_map[(row.buyer, row.seller)] = (row.transactions, row.quantity)
        rows = []
        for (node1, node2), (trans, quant) in txn_map.items():
            rows.append([node1, 'transaction', node2, trans, quant, quant])
        graph_edge = pd.DataFrame(rows, columns=['node1', 'label', 'node2', 'transactions', 'quantity', 'label;label'])
        return graph_edge


class Account:
    def __init__(self, org, assets):
        self.org = org
        self.txn = 0
        self.cash = 0
        self.quantity = 0
        self.portfolio = dict()
        for key in assets:
            self.portfolio[key] = 0
    def buy(self, asset, quantity, price):
        self.txn += 1
        self.cash -= quantity * price
        self.portfolio[asset] += quantity
        self.quantity += quantity
    def sell(self, asset, quantity, price):
        self.txn += 1
        self.cash += quantity * price
        self.portfolio[asset] -= quantity
        self.quantity += quantity
    def net(self):
        return sum([abs(x) for x in self.portfolio.values()])
    def __repr__(self):
        result = f'Account({self.org}, txn={self.txn}, quantity={self.quantity}, cash={self.cash}, {dict(self.portfolio)})'
        return result

class EdgeNode:
    def __init__(self, name, node1, node2, assets):
        self.name = name
        self.node1 = node1
        self.node2 = node2
        self.account = Account(name, assets)
    def buy(self, asset, quantity, price):
        self.account.buy(asset, quantity, price)
    def sell(self, asset, quantity, price):
        self.account.sell(asset, quantity, price)
    def __repr__(self):
        return self.account.__repr__()

class EdgeNodes:
    def __init__(self, name, assets, orgs=None):
        self.name = name
        self.assets = assets
        self.orgs = orgs
        self.edges = {}
    def add_transaction(self, buyer, seller, asset, quantity, price):
        if self.orgs:
            if not(buyer in self.orgs and seller in self.orgs):
                return
        if buyer < seller:
            edge_name = f'{buyer} x {seller}'
            if edge_name not in self.edges:
                self.edges[edge_name] = EdgeNode(edge_name, buyer, seller, self.assets)
            edge = self.edges[edge_name]
            edge.buy(asset, quantity, price)
        else:
            edge_name = f'{seller} x {buyer}'
            if edge_name not in self.edges:
                self.edges[edge_name] = EdgeNode(edge_name, seller, buyer, self.assets)
            edge = self.edges[edge_name]
            edge.sell(asset, quantity, price)


# class Transactions:
#     v = 3
#     def __init__(self, *, transaction: Transaction = None ):
#         self.txn_file = None
#         self.prices = defaultdict(list)
#         self.times = defaultdict(list)
#         self.quantity = defaultdict(list)
#         if transaction:
#             self.txn = transaction
#         else:
#             self.txn = Transaction()

#     def copy(self, other):
#         self.txn_file = other.txn_file
#         self.prices = other.prices
#         self.times = other.times
#         self.quanity = other.quantity
#         self.txn = other.txn

#     def load_file(self, txn_file):
#         '''Load transaction node tsv file'''
#         self.txn_file = txn_file
#         time_price = defaultdict(list)
#         prop = self.txn.p
#         with open(txn_file) as tsv_file:
#             reader = csv.DictReader(tsv_file, delimiter='\t')
#             for i, row in enumerate(reader):
#                 if i % 10000 == 9999:
#                     print('.', end='')
#                 price = float(row[prop.price])
#                 time = to_datetime(row[prop.time])
#                 quantity = float(row[prop.quantity])
#                 time_price[row[prop.asset]].append((time, price, quantity))
#             print()
#         for key in time_price:
#             result = sorted(time_price[key], key=itemgetter(0))
#             self.times[key] = [x[0] for x in result]
#             self.prices[key] = [x[1] for x in result]
#             self.quantity[key] = [x[2] for x in result]

#     def get_assts(self):
#         return list(self.prices.keys())

#     def get_price(self, asset: str, time: datetime.datetime) -> float:
#         index = bisect.bisect(self.times[asset], time)
#         if index >= len(self.prices[asset]):
#             index -= 1
#         return self.prices[asset][index]

#     def get_time_boundary(self) -> tuple:
#         min_time = min([times[0] for times in self.times.values()])
#         max_time = max([times[-1] for times in self.times.values()])
#         return (min_time, max_time)

class Transactions:
    def __init__(self, *, transaction: Transaction = None):
        self.txn_file = None
        self.txns = None
        if transaction:
            self.txn = transaction
        else:
            self.txn = Transaction()

    def copy(self, other):
        self.txn_file = other.txn_file
        self.prices = other.prices
        self.times = other.times
        self.quanity = other.quantity
        self.txn = other.txn

    def load_file(self, txn_file):
        '''Load transaction node tsv file'''
        self.txn_file = txn_file
        self.txns = pd.read_csv(txn_file, sep='\t')
        prop = self.txn.p
        self.txns[prop.time] = self.txns[prop.time].apply(to_datetime)

    def get_assets(self) -> list:
        prop = self.txn.p
        return self.txns.loc[:, prop.asset].drop_duplicates().to_list()

    def get_price(self, asset: str, time: datetime.datetime) -> float:
        prop = self.txn.p
        index = self.txns[self.txns[prop.asset] == asset][prop.time].searchsorted(time)
        if index >= self.txns.shape[0]:
            index -= 1
        return self.txns.loc[index, prop.price]

    def get_time_boundary(self) -> tuple:
        prop = self.txn.p
        min_time = self.txns[prop.time].min().to_pydatetime()
        max_time = self.txns[prop.time].max().to_pydatetime()
        return (min_time, max_time)

    def get_volume(self, asset: str = '', freq='1d'):
        prop = self.txn.p
        if asset:
            partial = self.txns[self.txns[prop.asset] == asset]
            return partial.groupby(pd.Grouper(key=prop.time, freq=freq))[prop.quantity].sum()
        return self.txns.groupby(pd.Grouper(key=prop.time, freq=freq))[prop.quantity].sum()


def transactions_to_tableau(txn_df, p: Transaction.Property, label_map: dict, dropped = None):
    results = pd.DataFrame(index=txn_df.index)
    results['transaction'] = txn_df.apply(lambda x: drop_en(x['label']), axis=1)
    results['buyer'] = txn_df.apply(lambda x: label_map[x[p.buyer]], axis=1)
    results['seller'] = txn_df.apply(lambda x: label_map[x[p.seller]], axis=1)
    results['time'] = txn_df.apply(lambda x: drop_precision(x[p.time]), axis=1)
    results['asset'] = 'symbol A'
    results['price'] = txn_df.apply(lambda x: x[p.price], axis=1)
    results['quantity'] = txn_df.apply(lambda x: x[p.quantity], axis=1)
    if dropped:
        results['dropped'] = dropped
    results

# dropped_transactions_file = '/home/ktyao/dev/finsec/IEM/output-fibo/simulation/trades/dropped_messages.tsv'
# dropped_transactions_raw = pd.read_csv(dropped_transactions_file, sep='\t')
# reconcile_time = datetime.datetime(2004, 10, 17, 20, 0, 0, 0)
# prop = txns.txn.p
# reconcile_prices = dict()
# for asset in txns.prices.keys():
#     reconcile_prices[asset] = txns.get_price(asset, reconcile_time)

# result = []
# for i, row in dropped_transactions_raw.iterrows():
#     txn_time = to_datetime(row[prop.time])
#     asset = row[prop.asset]
#     price = float(row[prop.price])
#     quantity = float(row[prop.quantity])
#     value = quantity * price
#     reconcile_price = reconcile_prices[asset]
#     print((price - reconcile_price) / price)
#     reconcile_value = quantity * reconcile_price
#     delta_value = value - reconcile_value
#     result.append([reconcile_time, reconcile_price, reconcile_value, delta_value])
# result_df = pd.DataFrame(result, columns=['reconcile_time', 'reconcile_price', 'reconcile_value', 'delta_value'])



# def change_in_value(dropped_transactions, txns, reconcile_time):
#     prop = txns.txn.p
#     reconcile_prices = dict()
#     for asset in txns.prices.keys():
#         reconcile_prices[asset] = txns.get_price(asset, reconcile_time)

#     result = []
#     for i, row in dropped_transactions.iterrows():
#         txn_time = to_datetime(row[prop.time])
#         asset = row[prop.asset]
#         price = float(row[prop.price])
#         quantity = float(row[prop.quantity])
#         value = quantity * price
#         reconcile_price = reconcile_prices[asset]
#         reconcile_value = quantity * reconcile_price
#         delta_value = value - reconcile_value
#         result.append([reconcile_time, reconcile_price, reconcile_value, value, delta_value])
#     result_df = pd.DataFrame(result, columns=['reconcile_time', 'reconcile_price', 'reconcile_value', 'value', 'delta_value'])
#     return result_df


def graph_org_nodes(accounts, shorten_label=True):
    rows = []
    for acc in accounts.values():
        tooltip = '</br>'.join([x for x in [
            f'person: {acc.org}', f'quantity: {acc.quantity}', f'cash: {acc.cash:.2f}']])
        label = acc.org
        if shorten_label:
            index = label.find(' (')
            if index > 0:
                label = label[0:index]
        label = en(label)
        row = [acc.org, label, 'org', '#1d52c4', acc.txn, acc.quantity, acc.cash, acc.net()] + list(acc.portfolio.values())
        row.append(tooltip)
        rows.append(row)
    columns = ['id', 'label', 'type', 'color', 'transactions', 'quantity', 'cash', 'net'] + list(acc.portfolio.keys()) + ['tooltip']
    return pd.DataFrame(rows, columns=columns)

def graph_edge_nodes(edge_nodes):
    edges = []
    rows = []
    for enode in edge_nodes.edges.values():
        ratio = enode.account.net()/enode.account.quantity
        tooltip = '</br>'.join([x for x in [
            f'transaction: {enode.name}',
            f'quantity:{enode.account.quantity}',
            f'net_change:{enode.account.net()}',
            f'ratio:{ratio:.3f}',
            f'cash:${enode.account.cash:.2f}']])
        # row = [enode.name, en(enode.name), 'edge_node', '#80148c', enode.account.txn,
        row = [enode.name, "''@en", 'edge_node', '#80148c', enode.account.txn,
               enode.account.quantity, enode.account.cash, enode.account.net(), ratio] + list(enode.account.portfolio.values())
        row.append(tooltip)
        rows.append(row)
        edges.append([enode.name, 'link', enode.node2, ratio])
        edges.append([enode.node1, 'link', enode.name, ratio])
    columns = ['id', 'label', 'type', 'color', 'transactions', 'quantity', 'cash', 'net', 'ratio'] + list(enode.account.portfolio.keys()) + ['tooltip']
    nodes = pd.DataFrame(rows, columns=columns)
    edges = pd.DataFrame(edges, columns=['node1', 'label', 'node2', 'ratio'])
    return nodes, edges

def filter_edge_nodes(edge_node_df, edge_df, value, column='quantity', keep_larger=True):
    if keep_larger:
        keep_node_df = edge_node_df[edge_node_df[column] >= value]
        drop_ids = set(edge_node_df[edge_node_df[column] < value]['id'].values)
    else:
        keep_node_df = edge_node_df[edge_node_df[column] < value]
        drop_ids = set(edge_node_df[edge_node_df[column] >= value]['id'].values)
    keep_edge_df = edge_df[(edge_df['node1'].isin(drop_ids) | edge_df['node2'].isin(drop_ids)) == False]
    return keep_node_df, keep_edge_df

def filter_singleton(node_df, edge_df):
    edge_nodes = set(edge_df['node1'].values).union(set(edge_df['node2'].values))
    return node_df[node_df['id'].isin(edge_nodes)]


def log_normalize(series):
    return np.log10(series - (series.min()-1))

def sqrt_normalize(series):
    return np.sqrt(series - series.min())

def scale(series, *, min_value=0, max_value=10):
    smin = series.min()
    smax = series.max()
    return (series-smin) * ((max_value - min_value) / (series.max()-series.min())) + min_value

def rbg_to_hex(rgb: list):
    return '#' + ''.join([f'{int(255*x):02x}' for x in rgb])

def rbg255_to_hex(rgb: list):
    return '#' + ''.join([f'{int(x):02x}' for x in rgb])

def hex_to_rgb(hex: str):
    if hex[0] == '#':
        hex = hex[1:]
    return [int(hex[i:i+2], 16) for i in range(0,6,2)]


def colors_red_green(series):
    rgb = scale(series, max_value=0.33).apply(lambda x: colorsys.hsv_to_rgb(x, 1.0, 1.0))
    return rgb.apply(rbg_to_hex)

# https://plotly.com/python/builtin-colorscales/
# px.colors.get_colorscale(name)
Inferno = ['#000004', '#1b0c41', '#4a0c6b', '#781c6d', '#a52c60', '#cf4446', '#ed6925', '#fb9b06', '#f7d13d', '#fcffa4']
Viridis = ['#440154', '#482878', '#3e4989', '#31688e', '#26828e', '#1f9e89', '#35b779', '#6ece58', '#b5de2b', '#fde725']
OrRd = ['#fff7ec', '#fee8c8', '#fdd49e', '#fdbb84', '#fc8d59', '#ef6548', '#d7301f', '#b30000', '#7f0000']
Gray = ['#000000', '#ffffff']

class ColorScale:
    def __init__(self, hex_colors: list, *, reverse=False, linspace_min=0, linspace_max=1):
        rgb_colors = [hex_to_rgb(x) for x in hex_colors]
        if reverse:
            rgb_colors.reverse()
        self.rgb_list = list(map(list, zip(*rgb_colors)))
        self.range = np.linspace(linspace_min, linspace_max, len(self.rgb_list[0]))
    def interp(self, x_values):
        results = []
        for color_list in self.rgb_list:
            results.append(np.interp(x_values, self.range, color_list))
        results = [rbg255_to_hex(x) for x in list(map(list, zip(*results)))]
        return results

def generate_transaction_graph(txn: pd.DataFrame):
    # Tableau transaction dataframe
    # columns = ['transaction', 'buyer', 'seller', 'time', 'asset', 'price', 'quantity']
    nodes = dict()
    edges = []
    for i, row in txn.iterrows():
        for prop in ['buyer', 'seller', 'time', 'asset', 'price', 'quantity']:
            edges.append([row['transaction'], prop, row[prop]])
        nodes[row['transaction']] = [row['transaction'], en(row['transaction']), 'transaction']
        if row['buyer'] not in nodes:
            nodes[row['buyer']] = [row['buyer'], en(row['buyer']), 'person']
        if row['seller'] not in nodes:
            nodes[row['seller']] = [row['seller'], en(row['seller']), 'person']
        if row['asset'] not in nodes:
            nodes[row['asset']] = [row['asset'], en(row['asset']), 'asset']
        for col in ['time', 'price', 'quantity']:
            if row[col] not in nodes:
                nodes[row[col]] = [row[col], en(str(row[col])), 'literal']
    edge_df = pd.DataFrame(edges, columns=['node1', 'label', 'node2'])
    node_df = pd.DataFrame(nodes.values(), columns=['id', 'label', 'type'])
    return edge_df, node_df


def generate_reconcile_graph(data: pd.DataFrame):
    # KG reconcile dataframe
    # columns = ['id', 'hasTransaction', 'holdsDuring', 'hasPrice', 'reconcileValue', 'transactionValue', 'deltaValue']
    nodes = dict()
    edges = []
    for i, row in data.iterrows():
        reconcile = ' '.join(row['id'].split('_')[1:])
        transaction = ' '.join(row['hasTransaction'].split('_')[1:])
        edges.append([reconcile, 'hasTransaction', transaction])
        for prop in ['holdsDuring', 'hasPrice', 'reconcileValue', 'transactionValue', 'deltaValue']:
            edges.append([reconcile, prop, row[prop]])
        nodes[reconcile] = [row['id'], en(row['id']), 'reconcile']
        for col in ['holdsDuring', 'hasPrice', 'reconcileValue', 'transactionValue', 'deltaValue']:
            if row[col] not in nodes:
                nodes[row[col]] = [row[col], en(str(row[col])), 'literal']
    edge_df = pd.DataFrame(edges, columns=['node1', 'label', 'node2'])
    node_df = pd.DataFrame(nodes.values(), columns=['id', 'label', 'type'])
    return edge_df, node_df
