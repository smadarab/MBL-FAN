
// import React, { createContext, useContext, useState, useEffect } from "react";
// import { auth, googleProvider } from "../firebase";
// import {
//     signInWithRedirect,
//     getRedirectResult,
//     signOut,
//     onAuthStateChanged,
// } from "firebase/auth";

// // Create the AuthContext
// const AuthContext = createContext();

// // AuthProvider Component
// export const AuthProvider = ({ children }) => {
//     const [user, setUser] = useState(null);
//     const [loading, setLoading] = useState(true);

//     // Sign In with Google
//     const googleSignIn = async () => {
//         try {
//             await signInWithRedirect(auth, googleProvider);
//         } catch (error) {
//             console.error("Error during Google Sign-In:", error.message);
//         }
//     };

//     // Sign Out
//     const logout = async () => {
//         try {
//             await signOut(auth);
//             console.log("User logged out.");
//         } catch (error) {
//             console.error("Error during logout:", error.message);
//         }
//     };

//     // Handle Redirect Results
//     useEffect(() => {
//         const handleRedirect = async () => {
//             try {
//                 const result = await getRedirectResult(auth);
//                 if (result) {
//                     console.log("User signed in through redirect:", result.user);
//                     setUser(result.user); // Set the user state
//                 }
//             } catch (error) {
//                 console.error("Error processing redirect result:", error.message);
//             }
//         };

//         handleRedirect();
//     }, []);

//     // Listen for Authentication State Changes
//     useEffect(() => {
//         const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
//             setUser(currentUser);
//             setLoading(false); // Set loading to false once the auth state is determined
//         });

//         return () => unsubscribe();
//     }, []);

//     return (
//         <AuthContext.Provider value={{ user, googleSignIn, logout, loading }}>
//             {children}
//         </AuthContext.Provider>
//     );
// };

// // Custom Hook for Using the Auth Context
// export const useAuth = () => {
//     return useContext(AuthContext);
// };



// AuthContext.jsx
import React, { createContext, useContext, useState, useEffect } from "react";
import {
    signInWithPopup,
    signOut,
    onAuthStateChanged
} from "firebase/auth";
import { doc, setDoc } from "firebase/firestore";
import { auth, db, googleProvider } from "../firebase";

// Create AuthContext
const AuthContext = createContext();

export function useAuth() {
    return useContext(AuthContext);
}

export function AuthProvider({ children }) {
    const [user, setUser] = useState(null);  // user state to store the current user
    const [loading, setLoading] = useState(true);  // loading state to track auth status

    // Google Sign In
    const signInWithGoogle = async () => {
        try {
            const result = await signInWithPopup(auth, googleProvider);
            const user = result.user;

            // Save user data to Firestore
            await setDoc(doc(db, "users", user.uid), {
                uid: user.uid,
                email: user.email,
                displayName: user.displayName,
                photoURL: user.photoURL,
                lastLogin: new Date().toISOString()
            }, { merge: true }); // merge: true will update existing documents instead of overwriting

            // Set user in context
            setUser(user);

            return user;
        } catch (error) {
            throw error;
        }
    };

    // Logout
    const logout = async () => {
        try {
            await signOut(auth);
            setUser(null);  // Reset the user when logged out
        } catch (error) {
            throw error;
        }
    };

    // Monitor auth state changes
    useEffect(() => {
        const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
            setUser(currentUser);  // Set user in context based on auth state
            setLoading(false);  // Set loading to false once auth state is determined
        });

        // Cleanup the listener when component unmounts
        return () => unsubscribe();
    }, []);

    // Provide the user, setUser, signInWithGoogle, logout to context
    const value = {
        user,
        setUser,  // Add setUser to context
        signInWithGoogle,
        logout
    };

    return (
        <AuthContext.Provider value={value}>
            {!loading && children}
        </AuthContext.Provider>
    );
}
