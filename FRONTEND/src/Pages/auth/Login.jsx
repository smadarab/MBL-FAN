import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../../context/authContext";
import AuthImage from "../../assets/Red-and-Black-Monogram-Sports-Baseball-Club-Logo.png";
import Button from "../../components/Button";


const Login = () => {
    const navigate = useNavigate();
    const { signInWithGoogle, user, setUser } = useAuth(); // Ensure setUser is available

    const [credentials, setCredentials] = useState({
        email: "",
        password: "",
    });

    const [error, setError] = useState("");

    useEffect(() => {
        if (user) {
            navigate("/get-start"); // Navigate if the user is authenticated
        }
    }, [user, navigate]);

    const handleChange = (e) => {
        const { name, value } = e.target;
        setCredentials({ ...credentials, [name]: value });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError("");

        try {
            const response = await fetch("http://35.206.83.91:8000/login", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    username: credentials.email,  // FastAPI expects "username"
                    password: credentials.password,
                }),
            });

            const data = await response.json();
            if (!response.ok) throw new Error(data.detail || "Login failed");

            alert(data.message);
            
            // Simulate storing user authentication details
            setUser({ email: credentials.email }); // Update user state in context

            navigate("/get-start"); // Navigate after successful login
        } catch (err) {
            setError(err.message);
        }
    };

    const handleGoogleSignIn = async () => {
        try {
            await signInWithGoogle();
        } catch (error) {
            console.error("Error signing in with Google:", error.message);
            setError("Google sign-in failed. Please try again.");
        }
    };

    return (
        <div
            style={{
                backgroundImage: `url(${AuthImage})`,
                backgroundSize: "cover",
                backgroundPosition: "center",
                backgroundRepeat: "no-repeat",
            }}
            className="h-screen w-full bg-black relative"
        >
            {/* Overlay */}
            <div className="absolute inset-0 bg-[#1B1B2E] bg-opacity-90"></div>

            {/* Content */}
            <div className="relative z-10 h-full flex flex-col justify-center items-center p-4 text-white">
                <h2 className="text-xl lg:text-3xl font-black uppercase mb-6">
                    Continue Playing Baseball Gaming Championship
                </h2>

                <div className="w-full max-w-lg bg-[#2C2C44] bg-opacity-70 p-6 rounded-lg border-4">
                    <h3 className="text-xl lg:text-2xl font-bold capitalize mb-4">Welcome Back to MBL</h3>

                    {error && <p className="text-red-500 mb-3">{error}</p>}

                    {/* Login Form */}
                    <form onSubmit={handleSubmit} className="flex flex-col gap-4">
                        <input
                            type="email"
                            name="email"
                            placeholder="Enter Your Email"
                            required
                            className="bg-transparent border p-3 rounded-lg text-white"
                            value={credentials.email}
                            onChange={handleChange}
                        />
                        <input
                            type="password"
                            name="password"
                            placeholder="Enter Your Password"
                            required
                            className="bg-transparent border p-3 rounded-lg text-white"
                            value={credentials.password}
                            onChange={handleChange}
                        />

                        <p>
                            Don't have an account?{" "}
                            <Link to="/register" className="underline hover:text-blue-400">
                                <i>Register</i>
                            </Link>
                        </p>

                        {/* Email/Password Login Button */}
                        <Button type="submit" className="border rounded-lg py-3 w-full text-sm lg:text-base">
                            Login
                        </Button>

                        {/* Google Sign-In Button */}
                        <button
                            type="button"
                            onClick={handleGoogleSignIn}
                            className="bg-blue-500 hover:bg-blue-600 text-white py-3 rounded-lg w-full"
                        >
                            Login with Google
                        </button>
                    </form>
                </div>
            </div>
        </div>
    );
};

export default Login;
