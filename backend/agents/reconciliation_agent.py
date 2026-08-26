from services.reconciliation import reconcile_entities


def reconciliation_agent(entities):
    return reconcile_entities(
        entities
    )