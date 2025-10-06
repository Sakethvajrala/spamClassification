chrome.action.onClicked.addListener(() => {
  let numberOfSpamEmails = 0;
  console.log("Extension icon clicked!");

  chrome.identity.getAuthToken({ interactive: true }, (token) => {
    if (chrome.runtime.lastError) {
      console.error("Auth error:", chrome.runtime.lastError);
      return;
    }

    if (!token) {
      console.error("No token returned!");
      return;
    }
    console.log("Got token:", token);

    // Step 1: List spam message IDs
    fetch("https://gmail.googleapis.com/gmail/v1/users/me/messages?q=in:spam", {
      headers: { Authorization: "Bearer " + token }
    })
      .then((res) => res.json())
      .then((data) => {
        if (!data.messages) {
          console.log("No spam emails found.");
          return;
        }

        console.log(`Found ${data.messages.length} spam emails`);

        // Step 2: Fetch details for each spam email
        data.messages.forEach((msg) => {
          fetch(
            `https://gmail.googleapis.com/gmail/v1/users/me/messages/${msg.id}?format=metadata&metadataHeaders=Subject&metadataHeaders=From`,
            {
              headers: { Authorization: "Bearer " + token }
            }
          )
            .then((res) => res.json())
            .then((fullMsg) => {
              const headers = fullMsg.payload.headers;
              const subject = headers.find(h => h.name === "Subject")?.value;
              const from = headers.find(h => h.name === "From")?.value;

              console.log(`📩 Spam from: ${from} | Subject: ${subject}`);
            })
            .catch((err) => console.error("Error fetching message details:", err));
        });
      })
      .catch((err) => console.error("Fetch error:", err));
  });
});1