import { createBrowserRouter, RouterProvider } from 'react-router-dom';
import './App.css';
import InputForm from "./components/InputForm.jsx";
import ResultPage from './components/ResultPage.jsx';

const appRouter = createBrowserRouter([
    {
        path: "/",
        element: <InputForm />
    },
    {
        path: "/result",
        element: <ResultPage />
    },
]);

const App = () => {
    return (
        <div className="flex flex-col relative justify-center items-center h-screen overflow-hidden bg-gradient-radial from-black to-black/80">

            {/* Fullscreen Background Video */}
            <div className="absolute -z-50 inset-0 w-full h-full ">
                <iframe
                    src="https://www.youtube.com/embed/Hgg7M3kSqyE?autoplay=1&loop=1&playlist=Hgg7M3kSqyE&controls=0&showinfo=0&modestbranding=1&rel=0&mute=1"
                    frameBorder="0"
                    className="w-full aspect-video"
                    allow="autoplay; encrypted-media"
                    referrerPolicy="strict-origin-when-cross-origin">
                </iframe>

            </div>

            {/* Your Main Content */}
            <RouterProvider router={appRouter}/>
        </div>
    );
};

export default App;
