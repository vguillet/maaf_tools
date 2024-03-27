from collections import defaultdict
from pprint import pformat, pprint

try:
    from maaf_tools.Singleton import Singleton
except:
    from maaf_tools.maaf_tools.Singleton import Singleton


class EventBus(metaclass=Singleton):

    def __init__(self):
        self.root_topic = Topic(["root"])

    @property
    def tree(self):
       tree = {}

       # -> Parse the tree
       def parse_tree(topic, tree):
           tree[topic.topic[-1]] = {
                "subs": list(topic.callbacks.keys()),
                "topics": {}
           }
           for subtopic in topic.subtopics.values():
               parse_tree(subtopic, tree[topic.topic[-1]])

       parse_tree(self.root_topic, tree)

       return tree

    def subscribe(self,
                  topic: list[str] or None,
                  subscriber_id: str,
                  callbacks: callable or list[callable]) -> None:
        """
        Add a callback to a topic, if it's None, it's added to the root topic.
        """

        topic = [] if topic is None else topic
        callbacks = callbacks if isinstance(callbacks, list) else [callbacks]

        pubsub_topic = self.search_add_topic(topic)
        pubsub_topic.add_subscriber(subscriber_id, callbacks)

    def remove_callback(self, callback: callable) -> None:
        self.root_topic.remove_callback(callback)

    def remove_subscriber(self, topic: list[str] or None, subscriber_id: str) -> None:
        topic = [] if topic is None else topic
        pubsub_topic = self.search_topic(topic)
        if pubsub_topic:
            pubsub_topic.remove_subscriber(subscriber_id)

    def get_callbacks(self, topic: list[str] or None) -> list[callable]:
        topic = [] if topic is None else topic

        callbacks = []
        pubsub_topic = self.search_topic(topic)
        if pubsub_topic:
            callbacks += pubsub_topic.get_callbacks()

        return callbacks

    def publish(self, topic: list[str], data: any) -> None:
        callbacks = self.get_callbacks(topic)
        for callback in callbacks:
            callback(data)

    def search_topic(self, keys: list[str]):
        active_topic = self.root_topic
        for key in keys:
            if key not in active_topic.subtopics:
                return None
            active_topic = active_topic.subtopics[key]
        return active_topic

    def search_add_topic(self, keys: list[str]):
        active_topic = self.root_topic
        for key in keys:
            active_topic = active_topic.subtopics[key]
        return active_topic


class TopicDict(dict):
    def __missing__(self, key):
        self[key] = Topic([key])
        return self[key]


class Topic:

    def __init__(self, topic: list[str]):
        self.topic = topic
        self.subtopics = TopicDict()
        self.callbacks = defaultdict(list)

    def add_subscriber(self,
                       subscriber_id: str,
                       callbacks: list[callable]) -> None:
        self.callbacks[subscriber_id] += callbacks

    def remove_subscriber(self, subscriber_id: str) -> None:

        self.callbacks.pop(subscriber_id, None)

        for subtopic in self.subtopics.values():
            subtopic.remove_subscriber(subscriber_id)

    def remove_callback(self, callback: callable) -> None:

        for sub, callbacks in self.callbacks.items():
            self.callbacks[sub] = [x for x in callbacks if x != callback]

        for subtopic in self.subtopics.values():
            subtopic.remove_subscriber(callback)

    def get_callbacks(self) -> list[callable]:
        callbacks = []

        for callback_list in self.callbacks.values():
            callbacks += callback_list

        for subtopic in self.subtopics.values():
            callbacks += subtopic.get_callbacks()

        return callbacks


if __name__ == "__main__":
    tree = EventBus()


    def hi(i):
        print(f"Hi: {i}")


    tree.subscribe(topic=["topic_1"], callbacks=hi, subscriber_id="sub_1")
    tree.subscribe(topic=[f"topic_1", "sub1"], callbacks=print, subscriber_id="sub1")
    tree.subscribe(topic=[f"topic_1", "sub2"], callbacks=hi, subscriber_id="sub1")
    tree.subscribe(topic=[f"topic_1", "sub1", "sub1"], callbacks=print, subscriber_id="sub1")
    tree.subscribe(topic=[f"topic_2", "sub1"], callbacks=hi, subscriber_id="sub1")
    tree.subscribe(topic=[f"topic_2"], callbacks=print, subscriber_id="sub1")

    # print(tree)
    # print(tree.get_subscribers(topic="topic_1"))

    tree.publish(topic=["topic_2"], data="hello")

    pprint(tree.tree, width=1, indent=1, compact=True, depth=2)