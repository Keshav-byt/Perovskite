import InputsA from "./inputs/InputsA.jsx";
import InputsMisc from "./inputs/InputsMisc.jsx";
import InputsB from "./inputs/InputsB.jsx";
import { useNavigate } from "react-router-dom";


const InputForm = () => {
    const navigate = useNavigate();
    const handleSubmit = () =>{
        navigate('/result')
    }
    return(
            <div className="input-box-border  backdrop-blur-lg text-white rounded-2xl p-4 flex flex-col items-center justify-center">
                <h1 className={'text-3xl font-bold  pl-8 m-2'}>Input Data for Band Gap determination</h1>
                <div className={' m-2 '}>
                    <div className="pl-8">
                        <h1 className={'mt-2'}>Functional Group of Element</h1>
                        <input type="text" placeholder={'Enter Functional Group of Element'}
                               className={'border-1 w-[50%] p-1'}/>
                    </div>
                    <div className='flex h-[60vh]'>
                        <InputsA/>
                        <InputsB/>
                        <InputsMisc/>
                    </div>
                    <div className={'flex flex-col items-center justify-center'}>
                        <button
                            onClick={() => handleSubmit()}
                            className='bg-white cursor-pointer text-black p-2 pl-4 pr-4 rounded-lg w-[10rem] font-bold mt-2 '>
                            SUBMIT
                        </button>

                    </div>
                </div>

            </div>

    )
}
export default InputForm