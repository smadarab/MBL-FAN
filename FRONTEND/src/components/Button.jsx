import ButtonSvg from "../assets/svg/ButtonSvg";

const Button = ({ className, href, onClick, children, px, white }) => {
    // to continue with my button

    const classes = `button relative inline-flex items-center 
  justify-center h-11 transition-colors 
  hover:text-color-1 ${px || "px-7"} ${white ? "text-n-8" : "text-n-1"} ${className || ""
        }`;

    const spanClasses = "relative z-10";

    const renderButton = () => (
        <button className={classes} onClick={onClick}>
            <span>{children}</span>
            {ButtonSvg(white)}
        </button>
    );

    const renderLink = () => (
        <a href={href} className={classes}>
            <span className={spanClasses}>{children}</span>
            {ButtonSvg(white)}
        </a>
    );

    //   and now when i have the renderLink and renderButton and then I chose which one
    return href ? renderLink() : renderButton();
};

export default Button;
