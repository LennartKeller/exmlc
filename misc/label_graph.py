import networkx as nx


class LabelGraphBuilder:
    """
    Builds a label where where labels which occur together are connected.
    If labels occurs together in multiple documents multiple edges are drawn.
    """

    def __init__(self):
        self.doc_2_id = {}
        self.label_set = []
        self.label_2_id = {}

    @staticmethod
    def __create_label_set(y):
        return list(set([tag for tags in y for tag in tags]))

    @staticmethod
    def __create_doc_2_id(X):
        return {x: index for index, x in enumerate(X)}

    def fit(self, X, y):
        """
        TODO docs
        1. Create label_set
        2. create doc_2_id mapping
        :param y:
        :return: self
        """

        self.label_set = self.__create_label_set(y)
        self.doc_2_id = self.__create_doc_2_id(X)
        return self

    def transform(self, X, y):
        """
        TODO docs
        Build the graph.
        :param X:
        :param y:
        :return:
        """
        if not self.label_set:
            raise Exception('Fit instance.')
        # TODO Raise error when instance is not fitted?

        graph = nx.MultiGraph()
        graph.add_nodes_from(self.label_set)

        for index, (xi, yi) in enumerate(zip(X, y)):
            if len(yi) > 0:
                first_label = yi[0]
                if len(yi) > 1:
                    for label in yi[1:]:
                        edge_data = graph.get_edge_data(first_label, label)
                        if edge_data:
                            already_edged = False
                            for key in edge_data.keys():
                                if edge_data[key]['doc_index'] == index:
                                    already_edged = True
                            if not already_edged:
                                graph.add_edge(first_label, label, doc_index=index, weight=1)
                        else:
                            graph.add_edge(first_label, label, doc_index=index, weight=1)
        return graph

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X, y)
