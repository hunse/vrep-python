if (sim_call_type==sim_childscriptcall_initialization) then
	cameraHandle=simGetObjectHandle('DVS128_sensor')
	angle=simGetScriptSimulationParameter(sim_handle_self,'cameraAngle')
	if (angle>100) then angle=100 end
	if (angle<34) then angle=34 end
	angle=angle*math.pi/180
	simSetObjectFloatParameter(cameraHandle,1004,angle)
	showConsole=simGetScriptSimulationParameter(sim_handle_self,'showConsole')
	if (showConsole) then
		auxConsole=simAuxiliaryConsoleOpen("DVS128 output",500,4)
	end

	showCameraView=simGetScriptSimulationParameter(sim_handle_self,'showCameraView')
	if (showCameraView) then
		floatingView=simFloatingViewAdd(0.2,0.8,0.4,0.4,0)
		simAdjustView(floatingView,cameraHandle,64)
	end

    -- clear event data
    data_signal = "currentDVS"
    simSetStringSignal(data_signal, '')
end

if (sim_call_type==sim_childscriptcall_cleanup) then
	if auxConsole then
		simAuxiliaryConsoleClose(auxConsole)
	end
end

if (sim_call_type==sim_childscriptcall_sensing) then

    -- clear data from last run of the script
	newData = ''

	if notFirstHere then
		r,t0,t1=simReadVisionSensor(cameraHandle)

		if (t1) then
			ts=math.floor(simGetSimulationTime()*1000)
			timeStampByte1=math.fmod(ts,256)
			timeStampByte2=(ts-timeStampByte1)/256
			for i=0,(#t1/3)-1,1 do
				byte1=math.floor(t1[3*i+2]+0.5)
				if (t1[3*i+1]>0) then
					byte1=byte1+128
				end
				byte2=math.floor(t1[3*i+3]+0.5)
				newData=newData..string.char(byte1)..string.char(byte2)..string.char(timeStampByte1)..string.char(timeStampByte2)

				if (showConsole) then
					if (t1[3*i+1]>0) then
						onOff=", on"
					else
						onOff=", off"
					end
					simAuxiliaryConsolePrint(auxConsole,"time="..ts.." ms, x="..math.floor(t1[3*i+2])..", y="..math.floor(t1[3*i+3])..onOff.."\n")
				end
			end
		end
	end
	notFirstHere=true

	-- newData now contains the same data as would the real sensor (i.e. for each pixel that changed:
	-- 7 bits for the x-coord, 1 bit for polatiry, 7 bits for the y-coord, 1 bit unused, and 1 word for the time stamp (in ms)
	-- You can access this data from outside via various mechanisms. For example:
	--
	--
	-- simSetStringSignal("dataFromThisTimeStep",newData)
    --
	-- Then in a different location:
	-- data=simGetStringSignal("dataFromThisTimeStep")
	--
	--
	-- Of course you can also send the data via tubes, wireless (simTubeOpen, etc., simSendData, etc.)
	--
	-- Also, if you you cannot read the data in each simulation
	-- step, then always append the data to an already existing signal data, e.g.
	--
	--
	-- existingData=simGetStringSignal("TheData")
	-- if existingData then
	--     data=existingData..data
	-- end
	-- simSetStringSignal("TheData",data)

	oldData = simGetStringSignal(data_signal)
	if oldData then
	    newData = oldData..newData
	end
	simSetStringSignal(data_signal, newData)
end
