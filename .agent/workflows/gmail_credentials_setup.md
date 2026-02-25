---
description: How to generate credentials.json for Gmail API
---

To use the Gmail fetcher, you need to provide a `credentials.json` file from the Google Cloud Console.

### Step 1: Create a Google Cloud Project
1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project (e.g., "Campus Hiring Dashboard").

### Step 2: Enable the Gmail API
1. Search for **"Gmail API"** in the search bar.
2. Click **Enable**.

### Step 3: Configure the OAuth Consent Screen
1. Go to **APIs & Services > OAuth consent screen**.
2. Select **External** and Click **Create**.
3. Fill in the **App name**, **User support email**, and **Developer contact info**.
4. Click **Save and Continue** until the end.

### Step 4: Create Credentials
1. Go to **APIs & Services > Credentials**.
2. Click **+ Create Credentials** and select **OAuth client ID**.
3. Select **Desktop app** as the Application type.
4. Name it "Campus Hiring" and click **Create**.

### Step 5: Download credentials.json
1. In the **OAuth 2.0 Client IDs** list, click the **Download icon** (down arrow) next to your client ID.
2. Rename the downloaded file to `credentials.json`.
3. Place it in the root directory of the project: `C:\Users\win10\.gemini\antigravity\scratch\Campus_Hiring_Dashboard\credentials.json`.
