import numpy as np


def extract_events(labels, unit_ids):
    labels = np.asarray(labels)
    unit_ids = np.asarray(unit_ids)
    events = []
    i, n = 0, len(labels)

    while i < n:
        if labels[i] == 1:
            j = i
            while j + 1 < n and labels[j + 1] == 1 and unit_ids[j + 1] == unit_ids[i]:
                j += 1
            events.append((int(unit_ids[i]), i, j))
            i = j + 1
        else:
            i += 1

    return events


def event_recall(events, preds):
    if not events:
        return float("nan")
    preds = np.asarray(preds)
    hits = sum(1 for _, s, e in events if preds[s:e + 1].any())
    return hits / len(events)


def mean_detection_delay(events, preds):
    preds = np.asarray(preds)
    delays = []
    for _, s, e in events:
        hit = np.where(preds[s:e + 1] == 1)[0]
        if len(hit) > 0:
            delays.append(int(hit[0]))
    return float(np.mean(delays)) if delays else float("nan")


def false_alarm_rate(preds, labels):
    preds, labels = np.asarray(preds), np.asarray(labels)
    normal = labels == 0
    if not normal.any():
        return float("nan")
    return float(preds[normal].mean())


def false_alarm_event_count(preds, labels, unit_ids):
    pred_events = extract_events(preds, unit_ids)
    true_events = extract_events(labels, unit_ids)

    def overlaps_any(s, e):
        return any(not (e < ts or te < s) for _, ts, te in true_events)

    return sum(1 for _, s, e in pred_events if not overlaps_any(s, e))
