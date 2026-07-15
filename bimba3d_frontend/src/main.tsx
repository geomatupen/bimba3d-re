import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  // GaussianSplats3D is not React.StrictMode-safe (double mount breaks the
  // worker lifecycle), so render the app without StrictMode for now.
  <BrowserRouter>
    <App />
  </BrowserRouter>
);
