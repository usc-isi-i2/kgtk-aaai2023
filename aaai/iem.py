
import math
import pandas as pd

from kgtk.functions import kgtk

from .transaction import Transaction
from .util import to_hour_precision, en, NODE_COLUMNS

txn = Transaction()
q = txn.Qnode()
p = txn.Property()


class IEM(Transaction):
    def __init__(self):
        super().__init__()

    def convert_row(self, row):
        time_precision = 15  # milli second
        financial_transaction = q.txn  # 'Q1166072'
        trade_id = f'QIEM_trade_{row["Trade_ID"]}'
        label = en(f'trade {row["Trade_ID"]}')
        description = en(f'IEM trade identifier {row["Trade_ID"]}')
        trade_date = str(row["Created_On"])
        trade_date = trade_date.replace(' ', 'T')
        if '.' not in trade_date:
            trade_date = trade_date + '.000000'
        date = f'^{trade_date}/{time_precision}'
        bundle = f'QIEM_bundle_{row["IsBundle"]}'
        asset = f'QIEM_asset_{row["Asset_ID"]}'
        market = f'QIEM_market_{row["Market_ID"]}'
        seller = f'QIEM_org_{row["Seller_ID"]}'
        seller_order = f'QIEM_seller_order_{row["SellerOrder_ID"]}'
        buyer = f'QIEM_org_{row["Buyer_ID"]}'
        buyer_order = f'QIEM_buyer_order_{row["BuyerOrder_ID"]}'
        quantity = row["Quantity"]
        price = row["Price"]
        return (trade_id, label, description, financial_transaction, date, bundle, asset, market,
                seller, seller_order, buyer, buyer_order, quantity, price)


    def kgtk_node_format(self, iem_df):
        trades = iem_df.apply(self.convert_row, axis=1, result_type='expand')
        trades.columns = ['id', 'label', 'description', p.isa, p.time,
                          p.bundle, p.asset, p.market, p.seller, p.sorder,
                          p.buyer, p.border, p.quantity, p.price]
        trades = trades.sort_values(p.time)
        return trades


    def edge_format(self, node_df, add_time=True, graph=''):
        columns = ['node1', 'label', 'node2', 'id']
        trade_edges = kgtk(node_df, "normalize-nodes / add-id --id-style wikidata")
        if add_time:
            columns = [p.time] + columns
            block_idx_map = {}
            for _, trade in node_df.iterrows():
                block_idx_map[trade.id] = to_hour_precision(trade[p.time])
            trade_edges[p.time] = trade_edges['node1'].apply(lambda x: block_idx_map[x])
            trade_edges = trade_edges.loc[:, [p.time, 'node1', 'label', 'node2', 'id']]
        if graph:
            columns = columns + [p.graph]
            trade_edges[p.graph] = graph
        trade_edges = trade_edges.loc[:, columns]
        return trade_edges

    def get_buyer_order_nodes(self, node_df):
        node_list = []
        for order in set(node_df[p.border].values):
            label = ' '.join(order.split('_')[1:])
            description = 'IEM ' + label
            node_list.append([order, label, description, math.nan, q.border])
        return pd.DataFrame(node_list, columns=NODE_COLUMNS + [p.isa])


    def get_seller_order_nodes(self, node_df):
        node_list = []
        for order in set(node_df[p.sorder].values):
            label = ' '.join(order.split('_')[1:])
            description = 'IEM ' + label
            node_list.append([order, label, description, math.nan, q.sorder])
        return pd.DataFrame(node_list, columns=NODE_COLUMNS + [p.isa])

    def get_asset_nodes(self, node_df):
        asset_df = pd.DataFrame(
            [['QIEM_asset_907', en('REP04'), en('Republican win'), math.nan, q.asset],
             ['QIEM_asset_906', en('DEM04'), en('Democratic win'), math.nan, q.asset],
             ['QIEM_asset_969', en('DEM04_L52'), en('Democratic win <52%'), math.nan, q.asset],
             ['QIEM_asset_968', en('DEM04_G52'), en('Democratic win >=52%'), math.nan, q.asset],
             ['QIEM_asset_971', en('REP04_L52'), en('Republican win <52%'), math.nan, q.asset],
             ['QIEM_asset_970', en('REP04_G52'), en('Republican win >=52%'), math.nan, q.asset]],
            columns=NODE_COLUMNS + [p.isa])
        return asset_df
        # node_list = []
        # for order in set(node_df[p.asset].values):
        #     label = ' '.join(order.split('_')[1:])
        #     description = 'IEM ' + label
        #     node_list.append([order, label, description, math.nan, q.asset])
        # return pd.DataFrame(node_list, columns=NODE_COLUMNS + [p.isa])

    def load_data(self, excel_file):
        '''Return IEM data in node format'''
        return self.kgtk_node_format(pd.read_excel(excel_file, dtype={
            'Trade_ID': int,
            'Asset_ID': int,
            'Market_ID': int,
            'Seller_ID': int,
            'SellerOrder_ID': int,
            'Buyer_ID': int,
            'BuyerOrder_ID': int,
            'Quantity': int}))
