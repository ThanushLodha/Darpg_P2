import { useState } from "react";
import Section from "./Section";
import { smallSphere, stars } from "../assets";
import Heading from "./Heading";
import axios from "axios";
import { toast, Toaster } from "sonner";


const Pricing = () => {
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [responseMessage, setResponseMessage] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setResponseMessage('');

    const id = toast('Classifying .... Hold On ... 🙏');
    toast.loading('Classifying .... Hold On ... 🙏', {
      id: id,
    });  
    try {
      
      await axios.post('http://localhost:8000/classify', { grievance: inputValue }).then((res) => setResponseMessage(res?.data?.department));
      toast.success("Success!");
    } catch (error) {
      setResponseMessage(JSON.stringify(error.message))
      toast.error("Error submitting grievance");
    } finally {
      setIsLoading(false);
      toast.dismiss(id);
    }
  };

  const handleChange = (e) => {
    setInputValue(e.target.value);
  };

  return (
    <>
    <Toaster/>
    <Section className="overflow-hidden" id="try-us">
      <div className="container relative z-2">
        <div className="hidden relative justify-center mb-[6.5rem] lg:flex">
          <img
            src={smallSphere}
            className="relative z-1"
            width={255}
            height={255}
            alt="Sphere"
          />
          <div className="absolute top-1/2 left-1/2 w-[60rem] -translate-x-1/2 -translate-y-1/2 pointer-events-none">
            <img
              src={stars}
              className="w-full"
              width={950}
              height={400}
              alt="Stars"
            />
          </div>
        </div>

        <Heading
          title="Enter Your Grievance !"
          className='text-center'
        />

        <form onSubmit={handleSubmit} className='flex flex-col items-center justify-center gap-9'>
          <textarea
            type="text"
            id="large-input"
            className="block w-full p-4 h-[100px] text-white border border-gray-300 rounded-lg text-base focus:border-none"
            value={inputValue}
            onChange={handleChange}
            required
          ></textarea>
          <button 
            className="button relative inline-flex items-center justify-center h-11 transition-colors hover:text-color-1 border-2 border-white border-opacity-15 p-3 rounded-lg" 
            type="submit"
            disabled={isLoading}
          >
            {isLoading ? "Submitting..." : "Submit"}
          </button>
        </form>

        {responseMessage && (
          <p className="text-gray-900 dark:text-white mt-4 text-center">{responseMessage}</p>
        )}
      </div>
    </Section>
    </>
  );
};

export default Pricing;
