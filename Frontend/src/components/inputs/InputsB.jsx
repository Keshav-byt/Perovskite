import data from "../../utils/columnHeadings.json";

const inputsB =()=>{
    return (
        <div className="p-4 m-4 overflow-y-scroll border-r-2 scrollbar-none border-gray-500">
            {data?.functional_groups[0]?.B &&
                Object.entries(data.functional_groups[0].B).map((value, key) => (
                    <div key={key} className={'mb-2'}>
                        <p>{value}</p>
                        <input placeholder="Enter Input" className={'border-1 p-1'}/>
                    </div>
                ))
            }
        </div>
    )
}
export default inputsB;