import { createContext, useContext } from "react";
import { api } from "./api";
import { useApi } from "./hooks";

// One /health poll shared by the sidebar status and every page that needs the decision threshold.
const ModelContext = createContext({ online: null, model: null, threshold: null });

export function ModelProvider({ children }) {
  const { data, error } = useApi(api.health, [], 15000);
  const value = {
    online: error ? false : data ? true : null,
    model: data?.model_loaded ? data.model : null,
    threshold: data?.model_loaded ? data.threshold : null,
  };
  return <ModelContext.Provider value={value}>{children}</ModelContext.Provider>;
}

export const useModel = () => useContext(ModelContext);
