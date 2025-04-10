import { useEffect, useState } from "react";
import { Outlet } from "react-router-dom";
function App() {
  const [message, setMessage] = useState("");

  useEffect(() => {
    fetch("/api/hello")
      .then((res) => res.json())
      .then((data) => setMessage(data.message));
  }, []);

  return (
    <div>
      <Outlet />
    </div>
  );
}

export default App;
