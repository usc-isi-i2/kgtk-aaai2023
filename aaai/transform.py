#!/usr/bin/env python

'''
Transform transaction knowledge graph to other types of knowledge
graphs, including trader graphs and trader-asset graphs.

'''

import argparse
import sys

from collections import defaultdict
from pathlib import Path

import pandas as pd
import sh

from kgtk.configure_kgtk_notebooks import ConfigureKGTK
from kgtk.functions import kgtk, kypher

import util

from transaction import Transaction

txn = Transaction()
p = txn.p
q = txn.q

def generate_dataset(transactions, item_id_map):
    item_len = len(item_id_map)

    def buyer_attribute_vector(x, state=0):
        result = [x['buyer'], x['asset'], x['time'], state]
        features = [0.0] * (3 * item_len)
        features[3 * x['asset']] = x['quantity']
        features[3 * x['asset'] + 1] = x['price']
        features[3 * x['asset'] + 2] = x['quantity'] * x['price']
        result = result + features
        return result

    def seller_attribute_vector(x, state=0):
        result = [x['seller'], x['asset'], x['time'], state]
        features = [0.0] * (3 * item_len)
        features[3 * x['asset']] = -x['quantity']
        features[3 * x['asset'] + 1] = x['price']
        features[3 * x['asset'] + 2] = -x['quantity'] * x['price']
        result = result + features
        return result

    data = pd.concat([
        transactions.apply(buyer_attribute_vector, axis=1, result_type='expand'),
        transactions.apply(seller_attribute_vector, axis=1, result_type='expand')],
                     ignore_index=True)
    data.columns = ['user_id', 'item_id', 'timestamp', 'state_label'] + list(range(3*item_len))
    data = data.astype({'user_id': 'int', 'item_id': 'int'})
    data = data.sort_values(by=['timestamp'])
    return data

'''
def to_csv(data_df, output_file):
    with open(output_file, 'w') as out:
        print('user_id,item_id,timestamp,state_label,comma_separated_list_of_features', file=out)
        for _, row in data_df.iterrows():
            result = []
            for i, x in enumerate(row.to_list()):
                if i < 2:
                    result.append(str(int(x)))
                else:
                    result.append(str(x))
            print(','.join(result), file=out)
'''

'''
def query_kgtk():
    assets = kypher(f"""
    -i claims -i label
    --match '
    claims: (asset)-[{p.isa}]->(:{q.asset}),
    label: (asset)-[:label]->(label)
    '
    --return 'asset as asset, kgtk_lqstring_text(label) as label'
    """)

    traders = kypher(f"""
    -i claims -i label
    --match '
    claims: (org)-[:{p.isa}]->(:{q.org}),
    label: (org)-[:label]->(label)
    '
    --return 'org as org, kgtk_lqstring_text(label) as label'
    """)

    transactions = kypher(f"""
    -i claims -i label
    --match '
    claims: (txn)-[:{p.isa}]->(:{q.txn}),
    claims: (txn)-[:{p.buyer}]->(buyer),
    claims: (txn)-[:{p.seller}]->(seller),
    claims: (txn)-[:{p.time}]->(time),
    claims: (txn)-[:{p.asset}]->(asset),
    claims: (txn)-[:{p.quantity}]->(quantity),
    claims: (txn)-[:{p.price}]->(price)'
    --return 'txn as id, buyer as buyer, seller as seller, time as time, asset as asset, quantity as quantity, price as price'
    """)

    return assets, traders, transactions
'''

def get_transactions(kgtk_file):
    transactions = kypher(f"""
    -i {kgtk_file}
    --match '
    (txn)-[:{p.isa}]->(:{q.txn}),
    (txn)-[:{p.buyer}]->(buyer),
    (txn)-[:{p.seller}]->(seller),
    (txn)-[:{p.time}]->(time),
    (txn)-[:{p.asset}]->(asset),
    (txn)-[:{p.quantity}]->(quantity),
    (txn)-[:{p.price}]->(price),
    (buyer)-[:label]->(buyer_label),
    (seller)-[:label]->(seller_label)'
    --return '
    txn as id, buyer as buyer, seller as seller, buyer_label as buyer_label,
    seller_label as seller_label, time as time, asset as asset, quantity as quantity, price as price'
    """)

    return transactions

def get_transactions_org(kgtk_file):
    transactions = kypher(f"""
    -i {kgtk_file}
    --match '
    (txn)-[:{p.isa}]->(:{q.txn}),
    (txn)-[:{p.buyer}]->(buyer),
    (txn)-[:{p.seller}]->(seller),
    (txn)-[:{p.time}]->(time),
    (txn)-[:{p.asset}]->(asset),
    (txn)-[:{p.quantity}]->(quantity),
    (txn)-[:{p.price}]->(price)'
    --return 'txn as id, buyer as buyer, seller as seller, time as time, asset as asset, quantity as quantity, price as price'
    """)

    return transactions

def to_trader_graph(transactions):
    seller_buyer = set()
    def add_pair(row):
        seller_buyer.add((row['seller'], 'soldTo', row['buyer']))
    transactions.apply(add_pair, axis=1)
    return pd.DataFrame(seller_buyer, columns=['node1', 'label', 'node2'])


def to_trader_graph_w_quantity(transactions):
    seller_buyer_quantity = defaultdict(int)
    seller_buyer_transaction = defaultdict(int)
    org_quantity = defaultdict(int)
    org_transaction = defaultdict(int)
    org_label = dict()
    org_txn_dates = defaultdict(list)
    org_date_stat = dict()
    def gather(row):
        seller_buyer_quantity[(row['seller'], row['buyer'])] += row['quantity']
        seller_buyer_transaction[(row['seller'], row['buyer'])] += 1
        org_quantity[row['seller']] += row['quantity']
        org_quantity[row['buyer']] += row['quantity']
        org_transaction[row['seller']] += 1
        org_transaction[row['buyer']] += 1
        org_label[row['seller']] = row['seller_label']
        org_label[row['buyer']] = row['buyer_label']
        date = util.to_datetime(row['time'])
        org_txn_dates[row['seller']].append(date)
        org_txn_dates[row['buyer']].append(date)

    transactions.apply(gather, axis=1)
    for org, dates in org_txn_dates.items():
        dates.sort()
        i = len(dates) // 2
        if len(dates) % 2 == 0:
            median = dates[i-1] + ((dates[i] - dates[i-1]) / 2)
        else:
            median = dates[i]
        org_date_stat[org] = [dates[0], median, dates[-1]]

    edges = pd.DataFrame(
        [(key[0], 'soldTo', key[1], value, seller_buyer_transaction[key])
         for key, value in seller_buyer_quantity.items()],
        columns=['node1', 'label', 'node2', 'quantity', 'transaction'])
    nodes = pd.DataFrame(
        [[node, label, org_quantity[node], org_transaction[node]] + org_date_stat[node]
         for node, label in org_label.items()],
        columns=['id', 'label', 'quantity', 'transaction', 'first_date', 'median_date', 'last_date'])
    return edges, nodes


def to_trader_asset_graph(transactions):
    trader_asset = set()
    def add_edges(row):
        trader_asset.add((row['seller'], 'sells', row['asset']))
        trader_asset.add((row['buyer'], 'buys', row['asset']))
    transactions.apply(add_edges, axis=1)
    return pd.DataFrame(trader_asset, columns=['node1', 'label', 'node2'])


'''
def configure_kgtk(kypher_dir, output_data_dir):
    kgtk_browser_dir = kypher_dir / 'browser'
    if not kgtk_browser_dir.exists():
        raise Exception(f'Directory does not exist: {kgtk_browser_dir}')

    tmp_dir = kypher_dir / 'tmp'
    project_name = kypher_dir.parts[-2]
    browser_files = [
        'labels.en.tsv.gz', 'aliases.en.tsv.gz', 'descriptions.en.tsv.gz', 'claims.tsv.gz',
        'claims.wikibase-item.tsv.gz', 'qualifiers.tsv.gz', 'metadata.property.datatypes.tsv.gz']
    kypher_files = [
        'label', 'claims', 'qualifiers']
    ck = ConfigureKGTK(kypher_files)
    ck.configure_kgtk(input_graph_path=str(kgtk_browser_dir),
                      output_path=str(tmp_dir),
                      project_name=project_name)
    ck.load_files_into_cache()
'''

def main(args):

    graph_type = args.graph
    if graph_type is None or graph_type not in output_graph_types:
        raise Exception(f'Invalid graph type: {graph_type}. Must be one of {", ".join(output_graph_types)}')

    input_file = args.i
    output_file = args.o
    output_node_file = args.n


    transactions = get_transactions(input_file)

    # if graph_type == 'trader':
    #     seller_buyer = to_trader_graph(transactions)
    #     seller_buyer = kgtk(seller_buyer, '''add-id''')

    #     seller_buyer.to_csv(output_file, sep='\t', index=False)
    if graph_type == 'trader':
        edges, nodes = to_trader_graph_w_quantity(transactions)
        # seller_buyer = kgtk(seller_buyer, '''add-id / normalize ''')

        edges.to_csv(output_file, sep='\t', index=False)
        if output_node_file:
            nodes.to_csv(output_node_file, sep='\t', index=False)

    elif graph_type == 'trader-asset':
        trader_asset = to_trader_asset_graph(transactions)
        trader_asset = kgtk(trader_asset, '''add-id''')

        trader_asset.to_csv(output_file, sep='\t', index=False)
    else:
        raise Exception(f'Graph type not implemented: {graph_type}')



# output_graph_types = ['trader', 'trader-w-quantity', 'trader-asset']
output_graph_types = ['trader', 'trader-asset']


# Example: python transform.py -i ~/dev/finsec/IEM/output_two/weekly/weekly.0.2004-06-11.tsv.gz -o ~/dev/finsec/IEM/output_two/weekly/trader-graph.weekly.0.2004-06-11.tsv.gz -g trader

if __name__ == '__main__':
    # print(sys.argv)
    parser = argparse.ArgumentParser(
        description='Transform KGTK transaction knowledge graph to types of knowledge graphs')
    parser.add_argument(
        '-i', metavar='INPUT_FILE', required=True, help='input file')
    parser.add_argument(
        '-o', metavar='OUTPUT_EDGE_FILE', required=True, help='output edge file')
    parser.add_argument(
        '-n', metavar='OUTPUT_NODE_FILE', help='output node file')
    # parser.add_argument(
    #     '-k', '--kypher-dir', metavar='OUTPUT_DIR', help='output directory for kypher')
    parser.add_argument(
        '-g', '--graph', metavar='GRAPH_TYPE', required=True, choices=output_graph_types,
        help=f'Output graph format. Possible values are: {", ".join(output_graph_types)}')
    parser.add_argument(
        '-v', action='store_true', help='verbose output')

    args = parser.parse_args()

    if args.v:
        bash_command = 'bash'
        sh_bash = sh.Command(bash_command)
        r = sh_bash('-c', 'echo $KGTK_GRAPH_CACHE')
        if len(r.stdout) == 1:
            print('Enviornment variable KGTK_GRAPH_CACHE is not set. Using default directory.')
        else:
            value = r.stdout.decode('utf-8').strip()
            print('Enviornment variable KGTK_GRAPH_CACHE={value}')

    # print(args)
    main(args)
