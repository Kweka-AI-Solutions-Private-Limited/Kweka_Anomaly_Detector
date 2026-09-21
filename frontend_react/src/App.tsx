import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AppLayout } from './components/layout/AppLayout';
import { Home } from './pages/Home';
import { Dashboard } from './pages/Dashboard';
import { Models } from './pages/Models';
import { ModelDetails } from './pages/ModelDetails';
import { CreateModel } from './pages/CreateModel';
import { Inspection } from './pages/Inspection';
import { InspectionHistory } from './pages/InspectionHistory';
import { InspectionDetail } from './pages/InspectionDetail';
import { FeedbackPage } from './pages/FeedbackPage';
import { Settings } from './pages/Settings';
import { LeafDiseasePage } from './pages/LeafDiseasePage';

export const App: React.FC = () => {
  return (
    <Router future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <Routes>
        <Route element={<AppLayout />}>
          <Route path="/" element={<Home />} />
          <Route path="/leaf-disease" element={<LeafDiseasePage />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/models" element={<Models />} />
          <Route path="/models/create" element={<CreateModel />} />
          <Route path="/models/:modelId" element={<ModelDetails />} />
          <Route path="/models/:modelId/build" element={<CreateModel />} />
          <Route path="/inspect" element={<Inspection />} />
          <Route path="/history" element={<InspectionHistory />} />
          <Route path="/inspections/:inspectionId" element={<InspectionDetail />} />
          <Route path="/feedback" element={<FeedbackPage />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </Router>
  );
};

export default App;
