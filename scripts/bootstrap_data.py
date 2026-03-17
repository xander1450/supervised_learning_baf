from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"


BILLING_TEXTS = [
    "I was charged twice for the same purchase",
    "Please refund the duplicate payment on my card",
    "My invoice amount is higher than what I approved",
    "I need help with a billing error on my subscription",
    "The payment went through but the receipt is missing",
    "Why was I billed again after cancelling",
    "There is an unauthorized charge on my account",
    "Can you reverse the extra transaction from yesterday",
    "I need a GST invoice for the amount I paid",
    "The discount code was ignored and I paid full price",
    "I got debited but the order did not complete",
    "Please send the final invoice for accounting",
    "The refund has not shown up on my credit card",
    "I was charged in the wrong currency",
    "There is tax added twice on this payment",
    "The payment page failed but money was deducted",
    "Please correct the invoice name and billing address",
    "The wallet credit was not applied at checkout",
    "I need a copy of the receipt for reimbursement",
    "My annual plan renewed unexpectedly",
    "The card was charged even though the trial should be free",
    "I cancelled last week but got billed again today",
    "There is a payment mismatch between the receipt and bank statement",
    "My company needs a revised invoice with VAT details",
]

BUG_TEXTS = [
    "The app crashes every time I try to upload a file",
    "Login fails with a blank screen after entering my password",
    "The dashboard keeps loading forever on Chrome",
    "Search results disappear when I apply a filter",
    "The export button does nothing when I click it",
    "I get a 500 error when opening the reports page",
    "Notifications are not showing up in the mobile app",
    "The file preview is broken for PDF documents",
    "The page freezes when I open settings",
    "I cannot reset my password because the link loops back",
    "The data table is rendering duplicate rows",
    "The checkout form is stuck on the final step",
    "The Android app closes immediately after launch",
    "The API responds with a timeout on save",
    "Attachments fail to download from the conversation screen",
    "The image upload spinner never stops",
    "I see a permission error even though I am an admin",
    "The filters reset automatically after every refresh",
    "The profile page shows stale data after update",
    "The delete action succeeds but the item stays visible",
    "The date picker selects the wrong timezone",
    "The website layout breaks on Safari",
    "The session expires too quickly while I am typing",
    "The app shows a network error despite stable internet",
]

FEATURE_REQUEST_TEXTS = [
    "Please add dark mode to the dashboard",
    "Can you support export to PDF for reports",
    "We need bulk edit for tags and categories",
    "It would help to have Slack notifications for approvals",
    "Please add role based access for the admin panel",
    "I want a keyboard shortcut for quick search",
    "Can you add multi language support in the app",
    "Please support scheduled reports every Monday",
    "It would be great to pin favorite dashboards",
    "We need an audit log for changes made by users",
    "Please add an option to duplicate templates",
    "Can you support SSO with Okta",
    "Please allow CSV import for contacts",
    "Add the ability to save filter presets",
    "It would be useful to compare reports side by side",
    "Please add webhooks for order status updates",
    "Can you support custom fields on the ticket form",
    "We want comments on shared dashboards",
    "Please add archive instead of permanent delete",
    "A mobile widget for quick actions would be helpful",
    "Can you add an API endpoint for analytics exports",
    "Please allow custom branding on generated reports",
    "It would help to have duplicate detection while importing",
    "We need a public share link with expiry controls",
]

NEW_LABELED_ROWS = [
    {"id": "new-001", "text": "Please refund the second charge from this morning", "label": "billing"},
    {"id": "new-002", "text": "The iPhone app crashes when I open notifications", "label": "bug"},
    {"id": "new-003", "text": "Please add recurring exports to Google Sheets", "label": "feature_request"},
    {"id": "new-004", "text": "I need a corrected invoice with the company GST number", "label": "billing"},
    {"id": "new-005", "text": "The save button hangs and the page never finishes loading", "label": "bug"},
    {"id": "new-006", "text": "Please add approval workflows for content publishing", "label": "feature_request"},
]



def build_dataset() -> pd.DataFrame:
    rows: list[dict] = []
    counter = 1

    for label, texts in [
        ("billing", BILLING_TEXTS),
        ("bug", BUG_TEXTS),
        ("feature_request", FEATURE_REQUEST_TEXTS),
    ]:
        for text in texts:
            rows.append({"id": f"train-{counter:03d}", "text": text, "label": label})
            counter += 1

    return pd.DataFrame(rows)


if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    train_df = build_dataset()
    new_df = pd.DataFrame(NEW_LABELED_ROWS)

    train_path = RAW_DIR / "train.csv"
    new_path = RAW_DIR / "new_labeled.csv"

    train_df.to_csv(train_path, index=False)
    new_df.to_csv(new_path, index=False)

    print(f"Wrote {len(train_df)} rows to {train_path}")
    print(f"Wrote {len(new_df)} rows to {new_path}")
