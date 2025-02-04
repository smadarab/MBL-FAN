import { useState, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../../context/authContext";
import AuthImage from "../../assets/Red-and-Black-Monogram-Sports-Baseball-Club-Logo.png";
import Button from "../../components/Button";

const Signup = () => {
    const { signInWithGoogle, user } = useAuth();
    const navigate = useNavigate();

    const [formData, setFormData] = useState({
        first_name: "",
        last_name: "",
        email: "",
        phone: "",
        password: "",
        confirmPassword: "",
        country: "",
    });

    const [error, setError] = useState("");

    useEffect(() => {
        if (user) {
            navigate('/get-start');
        }
    }, [user, navigate]);

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData({ ...formData, [name]: value });
    };

    const handleRegister = async (e) => {
        e.preventDefault();
        setError("");

        if (formData.password !== formData.confirmPassword) {
            setError("Passwords do not match.");
            return;
        }

        try {
            const response = await fetch("http://35.206.83.91:8000/register", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    username: formData.email, // FastAPI expects "username"
                    first_name: formData.first_name,
                    last_name: formData.last_name,
                    phone: formData.phone,
                    password: formData.password,
                    country: formData.country,
                }),
            });

            const data = await response.json();
            if (!response.ok) throw new Error(data.detail || "Registration failed");

            alert(data.message);
            navigate("/login");
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
            className="min-h-screen w-full bg-black relative"
        >
            {/* Overlay */}
            <div className="absolute inset-0 bg-[#1B1B2E] bg-opacity-90"></div>

            {/* Content */}
            <div className="relative z-10 min-h-screen flex flex-col justify-center items-center px-4 lg:px-10 text-white py-5 lg:py-10">
                <h2 className="text-xl mb-4 text-center lg:text-3xl lg:mb-8 font-black uppercase">
                    Start your journey with the Baseball Gaming Championship
                </h2>

                <div className="w-full max-w-lg lg:max-w-2xl bg-[#2C2C44] bg-opacity-70 p-6 rounded-lg border-4">
                    <h3 className="text-xl lg:text-2xl font-bold capitalize mb-4">
                        Register to start playing with friends
                    </h3>

                    {error && <p className="text-red-500 mb-3">{error}</p>}

                    {/* Signup Form */}
                    <form onSubmit={handleRegister} className="flex flex-col gap-4">
                        <input name="first_name" type="text" placeholder="First Name" required className="bg-transparent border p-3 rounded-lg text-white" onChange={handleChange} />
                        <input name="last_name" type="text" placeholder="Last Name" required className="bg-transparent border p-3 rounded-lg text-white" onChange={handleChange} />
                        <input name="email" type="email" placeholder="Email" required className="bg-transparent border p-3 rounded-lg text-white" onChange={handleChange} />
                        <input name="phone" type="tel" placeholder="Phone Number" required className="bg-transparent border p-3 rounded-lg text-white" onChange={handleChange} />
                        <input name="password" type="password" placeholder="Password" required className="bg-transparent border p-3 rounded-lg text-white" onChange={handleChange} />
                        <input name="confirmPassword" type="password" placeholder="Confirm Password" required className="bg-transparent border p-3 rounded-lg text-white" onChange={handleChange} />
                        <input name="country" type="text" placeholder="Country" required className="bg-transparent border p-3 rounded-lg text-white" onChange={handleChange} />

                        <p>
                            Already have an account?{" "}
                            <Link to="/login" className="underline hover:text-blue-400">
                                <i>Login</i>
                            </Link>
                        </p>

                        {/* Register Button */}
                        <Button type="submit" className="border rounded-lg py-3 w-full text-sm lg:text-base">
                            Sign Up
                        </Button>

                        {/* Google Sign-In Button */}
                        <button
                            type="button"
                            onClick={handleGoogleSignIn}
                            className="bg-blue-500 hover:bg-blue-600 text-white py-3 rounded-lg w-full"
                        >
                            Sign in with Google
                        </button>
                    </form>
                </div>
            </div>
        </div>
    );
};

export default Signup;
