import { useState, useEffect } from "react";

const CookieBanner = () => {
  const [showBanner, setShowBanner] = useState(false);

  useEffect(() => {
    const consent = localStorage.getItem("cookieConsent");
    if (!consent) {
      setShowBanner(true);
    }
  }, []);

  const handleAccept = () => {
    localStorage.setItem("cookieConsent", "true");
    setShowBanner(false);
  };

  if (!showBanner) return null;

  return (
    <div className="fixed bottom-0 left-0 w-full bg-gray-800 text-white p-4 z-50">
      <div className="max-w-4xl mx-auto flex flex-col md:flex-row justify-between items-center">
        <p className="text-sm md:text-base">
          We use cookies to improve your experience on our website. By
          continuing to use our site, you accept our{" "}
          <a
            href="/cookie-policy"
            className="text-blue-400 underline hover:text-blue-300"
            target="_blank"
            rel="noopener noreferrer"
          >
            Cookie Policy
          </a>
          .
        </p>
        <button
          onClick={handleAccept}
          className="bg-blue-600 text-white px-6 py-2 rounded-full mt-4 md:mt-0 shadow-md hover:bg-blue-500 transition"
        >
          Accept
        </button>
      </div>
    </div>
  );
};

export default CookieBanner;
