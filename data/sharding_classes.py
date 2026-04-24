class Client:
    def __init__(self, max_classes_per_client: int):
        self.max_classes = max_classes_per_client
        self.indices = []
        self.classes = []
    def add_index(self, new_index):
        self.indices.append(new_index)
    def add_classes(self, new_class):
        self.classes.append(new_class)
    def get_indices(self):
        return self.indices
    def get_classes(self):
        return self.classes
    def get_max_classes(self):
        return self.max_classes


class Class:
    def __init__(self, id_: int, max_clients_per_class: int):
        self.max_clients = max_clients_per_class
        self.id = id_
        self.client_list = []
        self.counter = 0
    def add_client(self, client: int):
        self.client_list.append(client)
    def update_counter(self):
        self.counter = (self.counter + 1) % self.max_clients
    def get_client_list(self):
        return self.client_list
    def get_id(self):
        return self.id
    def get_counter(self):
        return self.counter
    def get_max_clients(self):
        return self.max_clients


def find_available_client(clients: list[Client], class_: Class) -> tuple[int, bool]:
    if len(class_.get_client_list()) == class_.get_max_clients():   # class already assigned to max number of clients
        return class_.get_client_list()[class_.get_counter()], False    # return client from list of client assigned to current class
    for id_, client in enumerate(clients):
        if len(client.get_classes()) < client.get_max_classes() and class_.get_id() not in client.get_classes():    # if client can accomodate additional classes
            return id_, True
    print("error")
    return -2, False