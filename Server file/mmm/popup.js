document.getElementById('scrapeButton').addEventListener('click', function() {
  chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
    chrome.tabs.sendMessage(tabs[0].id, {action: 'scrapeContent'}, function(response) {
      document.getElementById('result').innerText = response.content;
    });
  });
});
