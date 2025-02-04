import { Navigate } from "react-router-dom";
import { useAuth } from "../../context/authContext";
import { auth } from "../../firebase";

const ProtectedRoute = ({ children }) => {
    const { user, loading } = useAuth();

    if (loading) {
        return <p>Loading...</p>;
    }

    // Allow access if either Firebase auth or API user auth is present
    if (!user && !auth.currentUser) {
        return <Navigate to="/login" replace />;
    }

    return children;
};

export default ProtectedRoute;
